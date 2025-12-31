"""Cost controls and quota management for the pipeline.

This module provides:
- Per-user job quotas based on subscription tier
- Daily/monthly usage tracking
- Cost estimation and attribution
- Rate limiting for free tier users
- Quota enforcement at job submission time

Usage:
    from blueprint_pipeline.quota import QuotaManager, UserTier

    quota = QuotaManager()

    # Check if user can submit a job
    if quota.check_user_quota(user_id):
        # Submit job
        queue.create_pipeline_task(...)
        quota.record_job_submission(user_id, capture_id)
    else:
        # Return quota exceeded error
        raise QuotaExceededError("Daily quota exceeded")
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check if Firestore is available
try:
    from google.cloud import firestore
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False
    firestore = None


class UserTier(str, Enum):
    """User subscription tiers with different quotas."""
    FREE = "free"
    PRO = "pro"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


@dataclass
class TierQuota:
    """Quota limits for a subscription tier."""
    tier: UserTier
    daily_jobs: int  # Jobs per day
    monthly_jobs: int  # Jobs per month
    max_video_duration_seconds: int  # Max video length
    max_concurrent_jobs: int  # Concurrent processing jobs
    priority_boost: bool  # Whether to use priority queue
    gpu_hours_per_month: float  # GPU compute hours included
    cost_per_extra_job: float  # Cost for jobs beyond quota

    @property
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "daily_jobs": self.daily_jobs,
            "monthly_jobs": self.monthly_jobs,
            "max_video_duration_seconds": self.max_video_duration_seconds,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "priority_boost": self.priority_boost,
            "gpu_hours_per_month": self.gpu_hours_per_month,
            "cost_per_extra_job": self.cost_per_extra_job,
        }


# Default tier quotas
TIER_QUOTAS = {
    UserTier.FREE: TierQuota(
        tier=UserTier.FREE,
        daily_jobs=3,
        monthly_jobs=30,
        max_video_duration_seconds=120,  # 2 minutes
        max_concurrent_jobs=1,
        priority_boost=False,
        gpu_hours_per_month=1.0,
        cost_per_extra_job=2.99,
    ),
    UserTier.PRO: TierQuota(
        tier=UserTier.PRO,
        daily_jobs=20,
        monthly_jobs=200,
        max_video_duration_seconds=600,  # 10 minutes
        max_concurrent_jobs=3,
        priority_boost=True,
        gpu_hours_per_month=10.0,
        cost_per_extra_job=1.99,
    ),
    UserTier.BUSINESS: TierQuota(
        tier=UserTier.BUSINESS,
        daily_jobs=100,
        monthly_jobs=1000,
        max_video_duration_seconds=1800,  # 30 minutes
        max_concurrent_jobs=10,
        priority_boost=True,
        gpu_hours_per_month=50.0,
        cost_per_extra_job=0.99,
    ),
    UserTier.ENTERPRISE: TierQuota(
        tier=UserTier.ENTERPRISE,
        daily_jobs=10000,  # Effectively unlimited
        monthly_jobs=100000,
        max_video_duration_seconds=7200,  # 2 hours
        max_concurrent_jobs=50,
        priority_boost=True,
        gpu_hours_per_month=500.0,
        cost_per_extra_job=0.49,
    ),
}


@dataclass
class UsageRecord:
    """Usage record for a user."""
    user_id: str
    tier: UserTier
    daily_jobs_used: int = 0
    monthly_jobs_used: int = 0
    concurrent_jobs: int = 0
    gpu_hours_used: float = 0.0
    last_job_time: Optional[datetime] = None
    daily_reset_time: Optional[datetime] = None
    monthly_reset_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "userId": self.user_id,
            "tier": self.tier.value,
            "dailyJobsUsed": self.daily_jobs_used,
            "monthlyJobsUsed": self.monthly_jobs_used,
            "concurrentJobs": self.concurrent_jobs,
            "gpuHoursUsed": self.gpu_hours_used,
            "lastJobTime": self.last_job_time.isoformat() if self.last_job_time else None,
            "dailyResetTime": self.daily_reset_time.isoformat() if self.daily_reset_time else None,
            "monthlyResetTime": self.monthly_reset_time.isoformat() if self.monthly_reset_time else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UsageRecord":
        return cls(
            user_id=data.get("userId", ""),
            tier=UserTier(data.get("tier", "free")),
            daily_jobs_used=data.get("dailyJobsUsed", 0),
            monthly_jobs_used=data.get("monthlyJobsUsed", 0),
            concurrent_jobs=data.get("concurrentJobs", 0),
            gpu_hours_used=data.get("gpuHoursUsed", 0.0),
            last_job_time=datetime.fromisoformat(data["lastJobTime"]) if data.get("lastJobTime") else None,
            daily_reset_time=datetime.fromisoformat(data["dailyResetTime"]) if data.get("dailyResetTime") else None,
            monthly_reset_time=datetime.fromisoformat(data["monthlyResetTime"]) if data.get("monthlyResetTime") else None,
        )


@dataclass
class QuotaCheckResult:
    """Result of a quota check."""
    allowed: bool
    reason: Optional[str] = None
    quota: Optional[TierQuota] = None
    usage: Optional[UsageRecord] = None
    remaining_daily: int = 0
    remaining_monthly: int = 0
    wait_time_seconds: Optional[int] = None  # For rate limiting


class QuotaExceededError(Exception):
    """Raised when user quota is exceeded."""
    def __init__(self, message: str, result: Optional[QuotaCheckResult] = None):
        super().__init__(message)
        self.result = result


class QuotaManager:
    """Manage user quotas and usage tracking.

    Stores usage data in Firestore under the `usage` collection.
    Each user has a document with their current usage stats.
    """

    COLLECTION = "usage"

    def __init__(self, project_id: Optional[str] = None):
        """Initialize quota manager.

        Args:
            project_id: GCP project ID (uses env var if not provided)
        """
        self._db = None
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PIPELINE_PROJECT_ID")

        if FIRESTORE_AVAILABLE:
            try:
                self._db = firestore.Client(project=self._project_id)
                logger.info(f"QuotaManager initialized for project: {self._project_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize Firestore: {e}")
        else:
            logger.warning("Firestore not available - quota tracking disabled")

    @property
    def is_available(self) -> bool:
        return self._db is not None

    def get_user_tier(self, user_id: str) -> UserTier:
        """Get user's subscription tier.

        Args:
            user_id: User ID

        Returns:
            User's subscription tier
        """
        if not self.is_available:
            return UserTier.FREE

        try:
            # Check users collection for subscription info
            user_doc = self._db.collection("users").document(user_id).get()

            if user_doc.exists:
                data = user_doc.to_dict()
                tier_str = data.get("subscriptionTier", "free")
                return UserTier(tier_str)

            return UserTier.FREE

        except Exception as e:
            logger.warning(f"Failed to get user tier: {e}")
            return UserTier.FREE

    def get_tier_quota(self, tier: UserTier) -> TierQuota:
        """Get quota limits for a tier.

        Args:
            tier: User tier

        Returns:
            TierQuota for the tier
        """
        return TIER_QUOTAS.get(tier, TIER_QUOTAS[UserTier.FREE])

    def get_usage(self, user_id: str) -> UsageRecord:
        """Get user's current usage.

        Args:
            user_id: User ID

        Returns:
            UsageRecord for the user
        """
        tier = self.get_user_tier(user_id)

        if not self.is_available:
            return UsageRecord(user_id=user_id, tier=tier)

        try:
            doc = self._db.collection(self.COLLECTION).document(user_id).get()

            if doc.exists:
                record = UsageRecord.from_dict(doc.to_dict())
                record = self._reset_if_needed(record)
                return record

            # Create new record
            return UsageRecord(
                user_id=user_id,
                tier=tier,
                daily_reset_time=self._get_next_daily_reset(),
                monthly_reset_time=self._get_next_monthly_reset(),
            )

        except Exception as e:
            logger.warning(f"Failed to get usage: {e}")
            return UsageRecord(user_id=user_id, tier=tier)

    def _reset_if_needed(self, record: UsageRecord) -> UsageRecord:
        """Reset counters if reset time has passed.

        Args:
            record: Current usage record

        Returns:
            Updated usage record
        """
        now = datetime.utcnow()

        # Check daily reset
        if record.daily_reset_time and now >= record.daily_reset_time:
            record.daily_jobs_used = 0
            record.daily_reset_time = self._get_next_daily_reset()

        # Check monthly reset
        if record.monthly_reset_time and now >= record.monthly_reset_time:
            record.monthly_jobs_used = 0
            record.gpu_hours_used = 0.0
            record.monthly_reset_time = self._get_next_monthly_reset()

        return record

    def _get_next_daily_reset(self) -> datetime:
        """Get next daily reset time (midnight UTC)."""
        now = datetime.utcnow()
        tomorrow = now + timedelta(days=1)
        return datetime(tomorrow.year, tomorrow.month, tomorrow.day)

    def _get_next_monthly_reset(self) -> datetime:
        """Get next monthly reset time (1st of next month UTC)."""
        now = datetime.utcnow()
        if now.month == 12:
            return datetime(now.year + 1, 1, 1)
        return datetime(now.year, now.month + 1, 1)

    def check_user_quota(
        self,
        user_id: str,
        video_duration_seconds: Optional[int] = None,
    ) -> QuotaCheckResult:
        """Check if user has quota available for a new job.

        Args:
            user_id: User ID
            video_duration_seconds: Optional video duration to check

        Returns:
            QuotaCheckResult with allowed status and details
        """
        tier = self.get_user_tier(user_id)
        quota = self.get_tier_quota(tier)
        usage = self.get_usage(user_id)

        # Check daily quota
        if usage.daily_jobs_used >= quota.daily_jobs:
            return QuotaCheckResult(
                allowed=False,
                reason=f"Daily quota exceeded ({usage.daily_jobs_used}/{quota.daily_jobs})",
                quota=quota,
                usage=usage,
                remaining_daily=0,
                remaining_monthly=max(0, quota.monthly_jobs - usage.monthly_jobs_used),
                wait_time_seconds=self._seconds_until_reset(usage.daily_reset_time),
            )

        # Check monthly quota
        if usage.monthly_jobs_used >= quota.monthly_jobs:
            return QuotaCheckResult(
                allowed=False,
                reason=f"Monthly quota exceeded ({usage.monthly_jobs_used}/{quota.monthly_jobs})",
                quota=quota,
                usage=usage,
                remaining_daily=0,
                remaining_monthly=0,
                wait_time_seconds=self._seconds_until_reset(usage.monthly_reset_time),
            )

        # Check concurrent jobs
        if usage.concurrent_jobs >= quota.max_concurrent_jobs:
            return QuotaCheckResult(
                allowed=False,
                reason=f"Maximum concurrent jobs reached ({usage.concurrent_jobs}/{quota.max_concurrent_jobs})",
                quota=quota,
                usage=usage,
                remaining_daily=quota.daily_jobs - usage.daily_jobs_used,
                remaining_monthly=quota.monthly_jobs - usage.monthly_jobs_used,
            )

        # Check video duration
        if video_duration_seconds and video_duration_seconds > quota.max_video_duration_seconds:
            return QuotaCheckResult(
                allowed=False,
                reason=f"Video too long ({video_duration_seconds}s > {quota.max_video_duration_seconds}s max)",
                quota=quota,
                usage=usage,
                remaining_daily=quota.daily_jobs - usage.daily_jobs_used,
                remaining_monthly=quota.monthly_jobs - usage.monthly_jobs_used,
            )

        # All checks passed
        return QuotaCheckResult(
            allowed=True,
            quota=quota,
            usage=usage,
            remaining_daily=quota.daily_jobs - usage.daily_jobs_used - 1,
            remaining_monthly=quota.monthly_jobs - usage.monthly_jobs_used - 1,
        )

    def _seconds_until_reset(self, reset_time: Optional[datetime]) -> Optional[int]:
        """Calculate seconds until reset time."""
        if not reset_time:
            return None
        delta = reset_time - datetime.utcnow()
        return max(0, int(delta.total_seconds()))

    def record_job_submission(
        self,
        user_id: str,
        capture_id: str,
        estimated_gpu_hours: float = 0.33,  # ~20 minutes
    ) -> bool:
        """Record a job submission for quota tracking.

        Args:
            user_id: User ID
            capture_id: Capture/job ID
            estimated_gpu_hours: Estimated GPU hours for the job

        Returns:
            True if recorded successfully
        """
        if not self.is_available:
            return False

        try:
            doc_ref = self._db.collection(self.COLLECTION).document(user_id)
            now = datetime.utcnow()

            # Use transaction to safely increment counters
            @firestore.transactional
            def update_in_transaction(transaction, doc_ref):
                doc = doc_ref.get(transaction=transaction)

                if doc.exists:
                    data = doc.to_dict()
                    record = UsageRecord.from_dict(data)
                    record = self._reset_if_needed(record)
                else:
                    tier = self.get_user_tier(user_id)
                    record = UsageRecord(
                        user_id=user_id,
                        tier=tier,
                        daily_reset_time=self._get_next_daily_reset(),
                        monthly_reset_time=self._get_next_monthly_reset(),
                    )

                # Increment counters
                record.daily_jobs_used += 1
                record.monthly_jobs_used += 1
                record.concurrent_jobs += 1
                record.gpu_hours_used += estimated_gpu_hours
                record.last_job_time = now

                # Save
                transaction.set(doc_ref, record.to_dict())

            transaction = self._db.transaction()
            update_in_transaction(transaction, doc_ref)

            logger.info(f"Recorded job submission for user {user_id}: {capture_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to record job submission: {e}")
            return False

    def record_job_completion(
        self,
        user_id: str,
        capture_id: str,
        actual_gpu_hours: Optional[float] = None,
    ) -> bool:
        """Record job completion (decrements concurrent jobs).

        Args:
            user_id: User ID
            capture_id: Capture/job ID
            actual_gpu_hours: Actual GPU hours used (for adjustment)

        Returns:
            True if recorded successfully
        """
        if not self.is_available:
            return False

        try:
            doc_ref = self._db.collection(self.COLLECTION).document(user_id)

            # Decrement concurrent jobs
            doc_ref.update({
                "concurrentJobs": firestore.Increment(-1),
            })

            # If actual GPU hours provided and differs from estimate, adjust
            if actual_gpu_hours is not None:
                # Could adjust gpu_hours_used if needed
                pass

            logger.info(f"Recorded job completion for user {user_id}: {capture_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to record job completion: {e}")
            return False

    def get_usage_summary(self, user_id: str) -> Dict[str, Any]:
        """Get usage summary for user.

        Args:
            user_id: User ID

        Returns:
            Summary dictionary
        """
        tier = self.get_user_tier(user_id)
        quota = self.get_tier_quota(tier)
        usage = self.get_usage(user_id)

        return {
            "user_id": user_id,
            "tier": tier.value,
            "quota": quota.to_dict,
            "usage": {
                "daily": {
                    "used": usage.daily_jobs_used,
                    "limit": quota.daily_jobs,
                    "remaining": max(0, quota.daily_jobs - usage.daily_jobs_used),
                    "reset_in_seconds": self._seconds_until_reset(usage.daily_reset_time),
                },
                "monthly": {
                    "used": usage.monthly_jobs_used,
                    "limit": quota.monthly_jobs,
                    "remaining": max(0, quota.monthly_jobs - usage.monthly_jobs_used),
                    "reset_in_seconds": self._seconds_until_reset(usage.monthly_reset_time),
                },
                "concurrent": {
                    "active": usage.concurrent_jobs,
                    "limit": quota.max_concurrent_jobs,
                },
                "gpu_hours": {
                    "used": usage.gpu_hours_used,
                    "included": quota.gpu_hours_per_month,
                    "remaining": max(0, quota.gpu_hours_per_month - usage.gpu_hours_used),
                },
            },
        }


# =============================================================================
# Cost Estimation
# =============================================================================

@dataclass
class CostEstimate:
    """Cost estimate for a pipeline job."""
    gpu_cost: float  # GPU compute cost
    storage_cost: float  # Storage cost
    egress_cost: float  # Network egress cost
    total_cost: float
    currency: str = "USD"
    breakdown: Dict[str, float] = field(default_factory=dict)


class CostEstimator:
    """Estimate costs for pipeline jobs."""

    # Cloud Run GPU pricing (us-central1)
    GPU_COST_PER_HOUR = 0.23  # NVIDIA L4
    CPU_COST_PER_HOUR = 0.024  # 1 vCPU
    MEMORY_COST_PER_GB_HOUR = 0.003  # per GB-hour

    # Storage pricing
    STORAGE_COST_PER_GB_MONTH = 0.020  # Standard class
    EGRESS_COST_PER_GB = 0.12  # NA to internet

    def estimate_job_cost(
        self,
        video_duration_seconds: int,
        resolution: tuple = (1920, 1080),
        include_storage: bool = True,
    ) -> CostEstimate:
        """Estimate cost for a pipeline job.

        Args:
            video_duration_seconds: Video duration in seconds
            resolution: Video resolution (width, height)
            include_storage: Whether to include storage costs

        Returns:
            CostEstimate with breakdown
        """
        # Estimate processing time based on video duration
        # Roughly 3-5x real-time for full pipeline
        processing_multiplier = 4.0
        processing_hours = (video_duration_seconds * processing_multiplier) / 3600

        # GPU cost
        gpu_cost = processing_hours * self.GPU_COST_PER_HOUR

        # CPU cost (4 vCPU)
        cpu_cost = processing_hours * self.CPU_COST_PER_HOUR * 4

        # Memory cost (16 GB)
        memory_cost = processing_hours * self.MEMORY_COST_PER_GB_HOUR * 16

        # Storage cost (estimate based on resolution and duration)
        # ~50MB per minute of video + ~100MB output
        video_size_gb = (video_duration_seconds / 60) * 0.05
        output_size_gb = 0.1  # Gaussian splat + poses
        storage_cost = (video_size_gb + output_size_gb) * self.STORAGE_COST_PER_GB_MONTH if include_storage else 0

        # Egress cost (downloading output)
        egress_cost = output_size_gb * self.EGRESS_COST_PER_GB

        total = gpu_cost + cpu_cost + memory_cost + storage_cost + egress_cost

        return CostEstimate(
            gpu_cost=gpu_cost + cpu_cost + memory_cost,
            storage_cost=storage_cost,
            egress_cost=egress_cost,
            total_cost=round(total, 4),
            breakdown={
                "gpu": round(gpu_cost, 4),
                "cpu": round(cpu_cost, 4),
                "memory": round(memory_cost, 4),
                "storage": round(storage_cost, 4),
                "egress": round(egress_cost, 4),
            },
        )

    def estimate_monthly_cost(
        self,
        jobs_per_day: int,
        avg_video_duration_seconds: int = 120,
    ) -> Dict[str, float]:
        """Estimate monthly cost for a given usage pattern.

        Args:
            jobs_per_day: Average jobs per day
            avg_video_duration_seconds: Average video duration

        Returns:
            Monthly cost estimate
        """
        job_cost = self.estimate_job_cost(avg_video_duration_seconds)
        monthly_jobs = jobs_per_day * 30

        return {
            "jobs_per_month": monthly_jobs,
            "cost_per_job": job_cost.total_cost,
            "monthly_compute": round(job_cost.gpu_cost * monthly_jobs, 2),
            "monthly_storage": round(job_cost.storage_cost * monthly_jobs, 2),
            "monthly_egress": round(job_cost.egress_cost * monthly_jobs, 2),
            "monthly_total": round(job_cost.total_cost * monthly_jobs, 2),
        }


# =============================================================================
# Utility Functions
# =============================================================================

def check_user_quota(user_id: str) -> bool:
    """Simple quota check for a user.

    Args:
        user_id: User ID

    Returns:
        True if user has quota available
    """
    manager = QuotaManager()
    result = manager.check_user_quota(user_id)
    return result.allowed


def get_user_tier(user_id: str) -> str:
    """Get user's subscription tier.

    Args:
        user_id: User ID

    Returns:
        Tier name as string
    """
    manager = QuotaManager()
    return manager.get_user_tier(user_id).value


def count_user_jobs_today(user_id: str) -> int:
    """Count user's jobs submitted today.

    Args:
        user_id: User ID

    Returns:
        Number of jobs submitted today
    """
    manager = QuotaManager()
    usage = manager.get_usage(user_id)
    return usage.daily_jobs_used


# Singleton instance
_quota_manager: Optional[QuotaManager] = None


def get_quota_manager() -> QuotaManager:
    """Get or create singleton QuotaManager instance."""
    global _quota_manager
    if _quota_manager is None:
        _quota_manager = QuotaManager()
    return _quota_manager
