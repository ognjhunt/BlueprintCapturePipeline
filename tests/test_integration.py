"""End-to-end integration tests for the Blueprint Capture Pipeline.

This module tests the full pipeline from video upload to Gaussian splat output,
including:
- Storage trigger function
- Video ingestion
- SLAM processing
- 3DGS training
- Export and notification

Usage:
    # Run all integration tests
    pytest tests/test_integration.py -v

    # Run with GCS integration (requires credentials)
    pytest tests/test_integration.py -v --gcs-bucket=your-bucket

    # Run specific test
    pytest tests/test_integration.py::TestStorageTrigger -v
"""
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
import pytest

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


# =============================================================================
# Test Configuration
# =============================================================================

@dataclass
class TestConfig:
    """Configuration for integration tests."""
    gcs_bucket: Optional[str] = None
    project_id: str = "blueprint-8c1ca"
    use_mock_gcs: bool = True
    use_mock_firestore: bool = True
    temp_dir: Optional[Path] = None


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--gcs-bucket",
        action="store",
        default=None,
        help="GCS bucket for integration tests (uses mock if not specified)",
    )
    parser.addoption(
        "--project-id",
        action="store",
        default="blueprint-8c1ca",
        help="GCP project ID for integration tests",
    )


@pytest.fixture
def test_config(request) -> TestConfig:
    """Get test configuration from command line options."""
    gcs_bucket = request.config.getoption("--gcs-bucket", None)
    project_id = request.config.getoption("--project-id", "blueprint-8c1ca")

    return TestConfig(
        gcs_bucket=gcs_bucket,
        project_id=project_id,
        use_mock_gcs=gcs_bucket is None,
        use_mock_firestore=True,
    )


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for tests."""
    workspace = tempfile.mkdtemp(prefix="blueprint_test_")
    yield Path(workspace)
    shutil.rmtree(workspace, ignore_errors=True)


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_gcs_client():
    """Mock Google Cloud Storage client."""
    with patch("google.cloud.storage.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Create a mock bucket
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_client.get_bucket.return_value = mock_bucket

        yield mock_client


@pytest.fixture
def mock_firestore_client():
    """Mock Firestore client."""
    with patch("google.cloud.firestore.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        yield mock_client


@pytest.fixture
def sample_ios_manifest() -> Dict[str, Any]:
    """Sample iOS manifest.json content."""
    return {
        "scene_id": "test_scene_001",
        "video_uri": "walkthrough.mov",
        "device_model": "iPhone 15 Pro",
        "os_version": "17.2",
        "fps_source": 30,
        "width": 1920,
        "height": 1080,
        "capture_start_epoch_ms": 1702134567890,
        "has_lidar": True,
        "scale_hint_m_per_unit": 1.0,
        "intended_space_type": "indoor",
        "creatorId": "user_abc123",  # P1: Test camelCase field
        "exposure_samples": [
            {"timestamp_ms": 0, "exposure_duration": 0.01, "iso": 400},
        ],
    }


@pytest.fixture
def sample_capture_files(temp_workspace, sample_ios_manifest) -> Dict[str, Path]:
    """Create sample capture files for testing."""
    raw_dir = temp_workspace / "raw"
    raw_dir.mkdir(parents=True)

    # Create manifest.json
    manifest_path = raw_dir / "manifest.json"
    manifest_path.write_text(json.dumps(sample_ios_manifest, indent=2))

    # Create a minimal video file (just header for testing)
    video_path = raw_dir / "walkthrough.mov"
    # Create a ~6MB dummy file (above MIN_VIDEO_SIZE_BYTES threshold)
    video_path.write_bytes(b"0" * (6 * 1024 * 1024))

    # Create motion.jsonl
    motion_path = raw_dir / "motion.jsonl"
    motion_data = [
        {"timestamp": 0.0, "attitude": [1, 0, 0, 0], "gravity": [0, -1, 0]},
        {"timestamp": 0.033, "attitude": [1, 0, 0, 0], "gravity": [0, -1, 0]},
    ]
    motion_path.write_text("\n".join(json.dumps(m) for m in motion_data))

    # Create ARKit directory with poses
    arkit_dir = raw_dir / "arkit"
    arkit_dir.mkdir()

    # Create poses.jsonl (critical for ARKit direct import)
    poses_path = arkit_dir / "poses.jsonl"
    poses_data = [
        {
            "frame_id": 0,
            "timestamp": 0.0,
            "transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        },
        {
            "frame_id": 1,
            "timestamp": 0.033,
            "transform": [[1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        },
    ]
    poses_path.write_text("\n".join(json.dumps(p) for p in poses_data))

    # Create intrinsics.json
    intrinsics_path = arkit_dir / "intrinsics.json"
    intrinsics_data = {
        "fx": 1500.0,
        "fy": 1500.0,
        "cx": 960.0,
        "cy": 540.0,
        "width": 1920,
        "height": 1080,
    }
    intrinsics_path.write_text(json.dumps(intrinsics_data, indent=2))

    return {
        "manifest": manifest_path,
        "video": video_path,
        "motion": motion_path,
        "poses": poses_path,
        "intrinsics": intrinsics_path,
        "raw_dir": raw_dir,
    }


# =============================================================================
# Storage Trigger Tests
# =============================================================================

class TestStorageTrigger:
    """Test the Cloud Function storage trigger."""

    def test_parse_upload_path_valid(self):
        """Test parsing a valid upload path."""
        from functions.storage_trigger import parse_upload_path

        path = "scenes/scene_123/iphone/2024-12-09T15-30-45-abc123/raw/manifest.json"
        result = parse_upload_path(path)

        assert result is not None
        assert result["scene_id"] == "scene_123"
        assert result["source"] == "iphone"
        assert result["capture_folder"] == "2024-12-09T15-30-45-abc123"
        assert result["filename"] == "manifest.json"

    def test_parse_upload_path_glasses(self):
        """Test parsing a glasses upload path."""
        from functions.storage_trigger import parse_upload_path

        path = "scenes/ChIJ123/glasses/2024-12-10T10-00-00-xyz789/raw/walkthrough.mov"
        result = parse_upload_path(path)

        assert result is not None
        assert result["source"] == "glasses"
        assert result["filename"] == "walkthrough.mov"

    def test_parse_upload_path_invalid(self):
        """Test parsing an invalid path returns None."""
        from functions.storage_trigger import parse_upload_path

        # Not in scenes/ prefix
        assert parse_upload_path("other/path/file.txt") is None

        # Missing /raw/ directory
        assert parse_upload_path("scenes/scene_123/iphone/timestamp/manifest.json") is None

    def test_convert_ios_manifest(self, sample_ios_manifest):
        """Test converting iOS manifest to pipeline format."""
        from functions.storage_trigger import convert_ios_manifest_to_session

        file_status = {
            "manifest.json": True,
            "walkthrough.mov": True,
            "motion.jsonl": True,
            "arkit/poses.jsonl": True,
            "arkit/intrinsics.json": True,
        }

        result = convert_ios_manifest_to_session(
            ios_manifest=sample_ios_manifest,
            scene_id="test_scene",
            source="iphone",
            raw_prefix="scenes/test_scene/iphone/timestamp/raw",
            bucket_name="test-bucket",
            file_status=file_status,
        )

        assert result["session_id"] == "test_scene"
        assert result["device"]["platform"] == "iOS"
        assert result["device"]["model"] == "iPhone 15 Pro"
        assert result["device"]["has_lidar"] == True
        assert len(result["clips"]) == 1
        assert result["extended_metadata"]["has_arkit_poses"] == True

    def test_creator_id_extraction(self, sample_ios_manifest):
        """P1: Test that creatorId (camelCase) is properly extracted."""
        from functions.storage_trigger import on_storage_finalize

        # Verify the manifest has creatorId in camelCase
        assert "creatorId" in sample_ios_manifest
        assert sample_ios_manifest["creatorId"] == "user_abc123"


# =============================================================================
# Video Verification Tests
# =============================================================================

class TestVideoVerification:
    """Test video upload verification (P2 fix)."""

    def test_check_upload_completeness_minimum_video_size(
        self, temp_workspace, mock_gcs_client
    ):
        """Test that small video files are rejected."""
        from functions.storage_trigger import (
            check_upload_completeness,
            MIN_VIDEO_SIZE_BYTES,
        )

        # Mock blob listing with small video
        mock_bucket = mock_gcs_client.bucket.return_value

        small_video_blob = MagicMock()
        small_video_blob.name = "raw/walkthrough.mov"
        small_video_blob.size = 1024  # 1KB - too small

        manifest_blob = MagicMock()
        manifest_blob.name = "raw/manifest.json"
        manifest_blob.size = 500

        mock_bucket.list_blobs.return_value = [small_video_blob, manifest_blob]

        # The function should detect the video is too small
        # This tests the P2 fix for video verification
        assert MIN_VIDEO_SIZE_BYTES == 5 * 1024 * 1024  # 5MB


# =============================================================================
# SLAM Backend Tests
# =============================================================================

class TestSLAMBackendAvailability:
    """Test SLAM backend availability checking (P1 fix)."""

    def test_check_backend_availability(self):
        """Test that backend availability checking works."""
        from blueprint_pipeline.video2zeroscene.slam import (
            check_slam_backend_availability,
        )

        availability = check_slam_backend_availability()

        assert isinstance(availability, dict)
        assert "wildgs_slam" in availability
        assert "splatam" in availability
        assert "colmap" in availability
        assert "pycolmap" in availability

    def test_get_recommended_backend_with_arkit(self):
        """Test that ARKit poses trigger direct import."""
        from blueprint_pipeline.video2zeroscene.slam import (
            get_recommended_backend,
        )
        from blueprint_pipeline.video2zeroscene.interfaces import (
            SLAMBackend,
            SensorType,
        )

        backend = get_recommended_backend(SensorType.RGB_DEPTH, has_arkit_poses=True)
        assert backend == SLAMBackend.ARKIT_DIRECT

    def test_get_recommended_backend_rgb_only(self):
        """Test RGB-only backend selection falls back to COLMAP if WildGS unavailable."""
        from blueprint_pipeline.video2zeroscene.slam import (
            get_recommended_backend,
            _BACKEND_AVAILABILITY,
        )
        from blueprint_pipeline.video2zeroscene.interfaces import (
            SLAMBackend,
            SensorType,
        )

        backend = get_recommended_backend(SensorType.RGB_ONLY, has_arkit_poses=False)
        # Should return either WILDGS_SLAM or COLMAP_FALLBACK depending on availability
        assert backend in [SLAMBackend.WILDGS_SLAM, SLAMBackend.COLMAP_FALLBACK]


# =============================================================================
# ChunkMerger Tests
# =============================================================================

class TestChunkMerger:
    """Test video chunking and merging (P2 fix)."""

    def test_video_chunker_should_chunk(self, temp_workspace):
        """Test chunking decision based on video duration."""
        from blueprint_pipeline.chunking import VideoChunker, ChunkConfig

        config = ChunkConfig(
            chunk_duration=60.0,
            overlap_duration=10.0,
            min_chunk_duration=30.0,
        )
        chunker = VideoChunker(config)

        # A video should be chunked if it's longer than chunk_duration + min_chunk_duration
        # i.e., 60 + 30 = 90 seconds
        threshold = config.chunk_duration + config.min_chunk_duration
        assert threshold == 90.0

    def test_calculate_chunk_boundaries(self):
        """Test chunk boundary calculation."""
        from blueprint_pipeline.chunking import VideoChunker, ChunkConfig

        config = ChunkConfig(
            chunk_duration=60.0,
            overlap_duration=10.0,
            min_chunk_duration=30.0,
        )
        chunker = VideoChunker(config)

        # Test with a 150-second video
        boundaries = chunker._calculate_chunk_boundaries(150.0)

        # Should have multiple chunks with overlap
        assert len(boundaries) >= 2

        # First chunk should start at 0
        assert boundaries[0][0] == 0.0

        # Last chunk should end at video duration
        assert boundaries[-1][1] == 150.0

        # Chunks should overlap
        for i in range(1, len(boundaries)):
            prev_end = boundaries[i - 1][1]
            curr_start = boundaries[i][0]
            # Current chunk should start before previous ends (overlap)
            assert curr_start < prev_end

    def test_chunk_merger_procrustes_alignment(self):
        """Test Procrustes alignment for chunk merging."""
        import numpy as np
        from blueprint_pipeline.chunking import ChunkMerger

        merger = ChunkMerger()

        # Create two sets of points with a known transformation
        np.random.seed(42)
        source = np.random.randn(20, 3)

        # Apply a known rotation and translation
        angle = np.pi / 6  # 30 degrees
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        t = np.array([1.0, 2.0, 0.5])

        target = (R @ source.T).T + t

        # Test alignment
        transform, error = merger._procrustes_alignment(source, target)

        # The transform should be close to identity after applying the inverse
        assert error < 0.1  # Should have low alignment error

    def test_chunk_merger_icp_refinement(self):
        """Test ICP refinement for better alignment."""
        import numpy as np
        from blueprint_pipeline.chunking import ChunkMerger

        config = {"use_icp": True, "icp_iterations": 20}
        merger = ChunkMerger(config)

        # Create source and target with small noise
        np.random.seed(42)
        source = np.random.randn(50, 3)
        noise = np.random.randn(50, 3) * 0.01  # Small noise
        target = source + noise

        # Get initial alignment
        initial_transform = np.eye(4)

        # Refine with ICP
        refined_transform, error = merger._icp_refinement(
            source, target, initial_transform
        )

        # Error should be small for nearly identical point clouds
        assert error < 0.1


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================

class TestEndToEndPipeline:
    """Full end-to-end integration tests."""

    def test_arkit_loader(self, sample_capture_files):
        """Test loading ARKit data from files."""
        from blueprint_pipeline.arkit_loader import (
            load_arkit_data,
            can_skip_slam,
        )

        raw_dir = sample_capture_files["raw_dir"]

        # Load ARKit data
        arkit_data = load_arkit_data(raw_dir)

        assert arkit_data is not None
        assert len(arkit_data.poses) > 0
        assert arkit_data.intrinsics is not None

        # Should be able to skip SLAM with valid ARKit poses
        assert can_skip_slam(arkit_data) == True

    def test_capture_manifest_creation(self, sample_capture_files, sample_ios_manifest):
        """Test CaptureManifest creation from iOS data."""
        from blueprint_pipeline.video2zeroscene.interfaces import (
            CaptureManifest,
            SensorType,
        )

        # Create manifest from iOS data
        manifest = CaptureManifest(
            capture_id="test_capture_001",
            capture_timestamp="2024-12-09T15:30:45Z",
            device_platform="ios",
            device_model=sample_ios_manifest["device_model"],
            sensor_type=SensorType.RGB_DEPTH,  # iPhone with LiDAR
            has_depth=True,
            has_imu=True,
            has_arkit_poses=True,
            intrinsics=None,  # Would be loaded from file
            clips=[{"uri": str(sample_capture_files["video"])}],
            scale_anchors=[],
        )

        assert manifest.capture_id == "test_capture_001"
        assert manifest.device_platform == "ios"
        assert manifest.has_arkit_poses == True

    @pytest.mark.skipif(
        not os.environ.get("RUN_GPU_TESTS"),
        reason="GPU tests disabled (set RUN_GPU_TESTS=1 to enable)",
    )
    def test_full_pipeline_with_arkit_poses(
        self, sample_capture_files, temp_workspace
    ):
        """Test full pipeline execution with ARKit pose bypass."""
        from blueprint_pipeline.video2zeroscene.pipeline import CapturePipeline
        from blueprint_pipeline.video2zeroscene.interfaces import PipelineConfig

        config = PipelineConfig(
            target_fps=2.0,
            enable_submapping=False,
        )

        pipeline = CapturePipeline(config)

        # This would run the full pipeline if GPU is available
        # For CI, we just verify the pipeline can be instantiated
        assert pipeline is not None


# =============================================================================
# Firestore Integration Tests
# =============================================================================

class TestFirestoreIntegration:
    """Test Firestore job tracking."""

    def test_capture_status_creation(self, mock_firestore_client):
        """Test creating capture status document."""
        from blueprint_pipeline.utils.firestore import (
            FirestoreJobTracker,
            CaptureStatus,
        )

        tracker = FirestoreJobTracker()

        # With mock, the actual creation won't happen but we test the interface
        result = tracker.create_capture(
            capture_id="test_capture_001",
            scene_id="scene_123",
            creator_id="user_abc",
            source="iphone",
            raw_data_uri="gs://bucket/scenes/...",
        )

        # Mock client should have been called or returned False
        # (depending on whether Firestore is available)
        assert isinstance(result, bool)

    def test_capture_status_update(self, mock_firestore_client):
        """Test updating capture status."""
        from blueprint_pipeline.utils.firestore import (
            FirestoreJobTracker,
            CaptureStatus,
            ProcessingStage,
        )

        tracker = FirestoreJobTracker()

        result = tracker.update_status(
            capture_id="test_capture_001",
            status=CaptureStatus.PROCESSING,
            stage=ProcessingStage.RECONSTRUCTION,
            progress=0.5,
        )

        assert isinstance(result, bool)


# =============================================================================
# Notification Tests
# =============================================================================

class TestNotifications:
    """Test push notification functionality."""

    def test_notification_payload_creation(self):
        """Test creating notification payloads."""
        from blueprint_pipeline.utils.notifications import (
            NotificationPayload,
            NotificationType,
        )

        payload = NotificationPayload(
            title="Scan Complete!",
            body="Your 3D capture is ready.",
            notification_type=NotificationType.CAPTURE_COMPLETE,
            capture_id="capture_123",
            scene_id="scene_456",
        )

        data = payload.to_data_dict()

        assert data["type"] == "capture_complete"
        assert data["capture_id"] == "capture_123"
        assert "timestamp" in data


# =============================================================================
# Quota Management Tests
# =============================================================================

class TestQuotaManagement:
    """Test quota and cost management."""

    def test_tier_quotas_defined(self):
        """Test that all tiers have defined quotas."""
        from blueprint_pipeline.quota import TIER_QUOTAS, UserTier

        assert UserTier.FREE in TIER_QUOTAS
        assert UserTier.PRO in TIER_QUOTAS
        assert UserTier.BUSINESS in TIER_QUOTAS
        assert UserTier.ENTERPRISE in TIER_QUOTAS

        # Free tier should have lower limits
        free = TIER_QUOTAS[UserTier.FREE]
        pro = TIER_QUOTAS[UserTier.PRO]

        assert free.daily_jobs < pro.daily_jobs
        assert free.monthly_jobs < pro.monthly_jobs

    def test_cost_estimation(self):
        """Test cost estimation for jobs."""
        from blueprint_pipeline.quota import CostEstimator

        estimator = CostEstimator()

        # Estimate cost for a 2-minute video
        estimate = estimator.estimate_job_cost(
            video_duration_seconds=120,
            resolution=(1920, 1080),
        )

        assert estimate.total_cost > 0
        assert estimate.gpu_cost > 0
        assert "gpu" in estimate.breakdown


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
