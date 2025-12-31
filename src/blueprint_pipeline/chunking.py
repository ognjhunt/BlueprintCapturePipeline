"""Video chunking for parallel processing of long videos.

This module provides functionality to split long videos into overlapping chunks
for parallel SLAM processing across multiple GPUs, then merge the results.

For videos > 5 minutes, splitting into chunks enables:
- Parallel processing on multiple GPUs
- Reduced memory requirements per GPU
- Better fault isolation (one chunk failure doesn't kill entire job)
- Faster total processing time

Usage:
    from blueprint_pipeline.chunking import VideoChunker, ChunkMerger

    # Split video into chunks
    chunker = VideoChunker(chunk_duration=60, overlap=10)
    chunks = chunker.create_chunks(video_path, output_dir)

    # Process chunks in parallel (via Cloud Tasks)
    for chunk in chunks:
        queue.create_pipeline_task(chunk_manifest_uri=chunk.manifest_uri)

    # Merge results after all chunks complete
    merger = ChunkMerger()
    merged_poses, merged_gaussians = merger.merge(chunk_results)
"""
from __future__ import annotations

import json
import logging
import math
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoChunk:
    """Represents a video chunk for processing."""
    chunk_id: int
    start_time: float  # seconds
    end_time: float  # seconds
    duration: float  # seconds
    overlap_start: float  # seconds of overlap with previous chunk
    overlap_end: float  # seconds of overlap with next chunk
    video_path: Path  # Path to extracted chunk video
    frame_offset: int  # Frame index offset in original video
    manifest_uri: Optional[str] = None  # GCS URI to chunk manifest

    @property
    def is_first(self) -> bool:
        return self.chunk_id == 0

    @property
    def has_overlap_with_previous(self) -> bool:
        return self.overlap_start > 0

    @property
    def has_overlap_with_next(self) -> bool:
        return self.overlap_end > 0


@dataclass
class ChunkConfig:
    """Configuration for video chunking."""
    chunk_duration: float = 60.0  # seconds per chunk
    overlap_duration: float = 10.0  # seconds of overlap between chunks
    min_chunk_duration: float = 30.0  # minimum chunk size (don't create tiny last chunk)
    max_chunks: int = 20  # maximum number of chunks
    target_fps: float = 30.0  # FPS for chunk extraction


@dataclass
class ChunkResult:
    """Result from processing a single chunk."""
    chunk_id: int
    success: bool
    poses: Optional[np.ndarray] = None  # Nx4x4 camera poses
    timestamps: Optional[np.ndarray] = None  # N timestamps
    gaussians_path: Optional[Path] = None
    point_cloud: Optional[np.ndarray] = None  # Nx3 points
    scale_factor: float = 1.0  # Scale relative to first chunk
    error: Optional[str] = None


@dataclass
class MergedResult:
    """Result from merging all chunks."""
    success: bool
    poses: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    gaussians_path: Optional[Path] = None
    total_frames: int = 0
    alignment_errors: List[float] = field(default_factory=list)
    error: Optional[str] = None


class VideoChunker:
    """Split long videos into overlapping chunks for parallel processing.

    The chunking strategy:
    1. Extract video metadata (duration, fps)
    2. Calculate chunk boundaries with overlap
    3. Extract each chunk as separate video file
    4. Create manifest for each chunk with timing info

    Overlap regions are used for:
    - Aligning poses between chunks (using feature matching)
    - Smooth transitions in merged Gaussian splat
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()

    def should_chunk(self, video_path: Path) -> bool:
        """Determine if video should be chunked based on duration.

        Args:
            video_path: Path to video file

        Returns:
            True if video should be chunked
        """
        duration = self._get_video_duration(video_path)
        # Chunk if video is longer than chunk_duration + min_chunk_duration
        # (ensures we don't create a tiny last chunk)
        threshold = self.config.chunk_duration + self.config.min_chunk_duration
        return duration > threshold

    def create_chunks(
        self,
        video_path: Path,
        output_dir: Path,
        session_id: str,
    ) -> List[VideoChunk]:
        """Split video into overlapping chunks.

        Args:
            video_path: Path to source video
            output_dir: Directory to write chunk videos
            session_id: Session ID for naming

        Returns:
            List of VideoChunk objects
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get video metadata
        duration = self._get_video_duration(video_path)
        fps = self._get_video_fps(video_path)

        # Calculate chunk boundaries
        chunks = self._calculate_chunk_boundaries(duration)

        if len(chunks) > self.config.max_chunks:
            logger.warning(f"Video requires {len(chunks)} chunks, exceeds max {self.config.max_chunks}")
            # Increase chunk duration to fit within max
            new_duration = duration / self.config.max_chunks + self.config.overlap_duration
            self.config.chunk_duration = new_duration
            chunks = self._calculate_chunk_boundaries(duration)

        logger.info(f"Splitting {duration:.1f}s video into {len(chunks)} chunks")

        # Extract each chunk
        result_chunks = []
        for i, (start, end) in enumerate(chunks):
            chunk_path = output_dir / f"chunk_{i:03d}.mp4"

            # Extract video segment
            success = self._extract_video_segment(
                video_path, chunk_path, start, end
            )

            if not success:
                logger.error(f"Failed to extract chunk {i}")
                continue

            # Calculate overlap durations
            overlap_start = 0.0
            overlap_end = 0.0

            if i > 0:
                prev_end = chunks[i - 1][1]
                overlap_start = prev_end - start

            if i < len(chunks) - 1:
                next_start = chunks[i + 1][0]
                overlap_end = end - next_start

            chunk = VideoChunk(
                chunk_id=i,
                start_time=start,
                end_time=end,
                duration=end - start,
                overlap_start=overlap_start,
                overlap_end=overlap_end,
                video_path=chunk_path,
                frame_offset=int(start * fps),
            )
            result_chunks.append(chunk)

        # Write chunk index
        self._write_chunk_index(result_chunks, output_dir, session_id)

        return result_chunks

    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds using ffprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return float(result.stdout.strip())
        except Exception as e:
            logger.error(f"Failed to get video duration: {e}")
            return 0.0

    def _get_video_fps(self, video_path: Path) -> float:
        """Get video FPS using ffprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=r_frame_rate",
                    "-of", "csv=p=0",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            fps_str = result.stdout.strip()
            if "/" in fps_str:
                num, den = fps_str.split("/")
                return float(num) / float(den)
            return float(fps_str)
        except Exception as e:
            logger.error(f"Failed to get video FPS: {e}")
            return 30.0

    def _calculate_chunk_boundaries(self, duration: float) -> List[Tuple[float, float]]:
        """Calculate start/end times for each chunk with overlap.

        Args:
            duration: Total video duration in seconds

        Returns:
            List of (start_time, end_time) tuples
        """
        chunks = []
        chunk_duration = self.config.chunk_duration
        overlap = self.config.overlap_duration
        effective_duration = chunk_duration - overlap

        current_start = 0.0

        while current_start < duration:
            # End time is start + chunk_duration, capped at video end
            end_time = min(current_start + chunk_duration, duration)

            # If remaining duration is less than min_chunk_duration,
            # extend previous chunk instead of creating tiny chunk
            remaining = duration - end_time
            if 0 < remaining < self.config.min_chunk_duration:
                end_time = duration

            chunks.append((current_start, end_time))

            # Next chunk starts at effective_duration after this one
            # (which creates overlap)
            current_start += effective_duration

            # Stop if we've reached the end
            if end_time >= duration:
                break

        return chunks

    def _extract_video_segment(
        self,
        input_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
    ) -> bool:
        """Extract a video segment using ffmpeg.

        Args:
            input_path: Source video path
            output_path: Output chunk path
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            True if extraction successful
        """
        duration = end_time - start_time

        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",  # Overwrite output
                    "-ss", str(start_time),  # Seek before input for speed
                    "-i", str(input_path),
                    "-t", str(duration),
                    "-c:v", "libx264",  # Re-encode for clean cuts
                    "-preset", "fast",
                    "-crf", "18",  # High quality
                    "-an",  # No audio
                    str(output_path),
                ],
                capture_output=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                logger.error(f"ffmpeg failed: {result.stderr.decode()}")
                return False

            return output_path.exists()

        except Exception as e:
            logger.error(f"Failed to extract video segment: {e}")
            return False

    def _write_chunk_index(
        self,
        chunks: List[VideoChunk],
        output_dir: Path,
        session_id: str,
    ) -> None:
        """Write chunk index file for tracking."""
        index = {
            "session_id": session_id,
            "total_chunks": len(chunks),
            "config": {
                "chunk_duration": self.config.chunk_duration,
                "overlap_duration": self.config.overlap_duration,
            },
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "start_time": c.start_time,
                    "end_time": c.end_time,
                    "duration": c.duration,
                    "overlap_start": c.overlap_start,
                    "overlap_end": c.overlap_end,
                    "video_path": str(c.video_path),
                    "frame_offset": c.frame_offset,
                }
                for c in chunks
            ],
        }

        index_path = output_dir / "chunk_index.json"
        index_path.write_text(json.dumps(index, indent=2))
        logger.info(f"Wrote chunk index to {index_path}")


class ChunkMerger:
    """Merge SLAM results from multiple chunks into a single consistent model.

    The merging process:
    1. Load poses and point clouds from each chunk
    2. Find correspondences in overlap regions using feature matching
    3. Compute rigid transforms to align chunks
    4. Apply transforms and merge point clouds
    5. Optionally refine with global bundle adjustment
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.alignment_threshold = self.config.get("alignment_threshold", 0.1)  # meters
        self.use_icp = self.config.get("use_icp", True)
        self.icp_iterations = self.config.get("icp_iterations", 50)

    def merge(
        self,
        chunk_results: List[ChunkResult],
        chunk_index: Dict[str, Any],
    ) -> MergedResult:
        """Merge chunk results into a single consistent model.

        Args:
            chunk_results: Results from each chunk (must be sorted by chunk_id)
            chunk_index: Chunk index with timing information

        Returns:
            MergedResult with aligned poses and merged point cloud
        """
        if not chunk_results:
            return MergedResult(success=False, error="No chunk results provided")

        # Filter successful chunks
        successful = [r for r in chunk_results if r.success and r.poses is not None]
        if not successful:
            return MergedResult(success=False, error="No successful chunk results")

        # Sort by chunk_id
        successful.sort(key=lambda x: x.chunk_id)

        # If only one chunk, return as-is
        if len(successful) == 1:
            return MergedResult(
                success=True,
                poses=successful[0].poses,
                timestamps=successful[0].timestamps,
                gaussians_path=successful[0].gaussians_path,
                total_frames=len(successful[0].poses),
            )

        try:
            # Align chunks sequentially
            aligned_poses = []
            aligned_timestamps = []
            alignment_errors = []

            # First chunk is reference (identity transform)
            reference = successful[0]
            aligned_poses.append(reference.poses)
            aligned_timestamps.append(reference.timestamps)

            cumulative_transform = np.eye(4)

            for i in range(1, len(successful)):
                current = successful[i]
                previous = successful[i - 1]

                # Get chunk info for overlap calculation
                current_chunk_info = chunk_index["chunks"][current.chunk_id]
                prev_chunk_info = chunk_index["chunks"][previous.chunk_id]

                # Find overlap region
                overlap_start = current_chunk_info["start_time"]
                overlap_end = prev_chunk_info["end_time"]
                overlap_duration = overlap_end - overlap_start

                # Get poses in overlap region
                prev_overlap_poses = self._get_poses_in_timerange(
                    previous.poses,
                    previous.timestamps,
                    overlap_start - prev_chunk_info["start_time"],
                    overlap_end - prev_chunk_info["start_time"],
                )

                curr_overlap_poses = self._get_poses_in_timerange(
                    current.poses,
                    current.timestamps,
                    0,
                    overlap_duration,
                )

                if len(prev_overlap_poses) == 0 or len(curr_overlap_poses) == 0:
                    logger.warning(f"No overlap poses found between chunk {i-1} and {i}")
                    # Use identity transform as fallback
                    transform = np.eye(4)
                    alignment_error = float("inf")
                else:
                    # Compute alignment transform
                    transform, alignment_error = self._compute_alignment_transform(
                        prev_overlap_poses,
                        curr_overlap_poses,
                    )
                    alignment_errors.append(alignment_error)

                # Update cumulative transform
                cumulative_transform = cumulative_transform @ transform

                # Apply transform to current chunk poses
                transformed_poses = self._apply_transform(
                    current.poses,
                    cumulative_transform,
                )

                # Remove overlap frames from current chunk
                non_overlap_mask = current.timestamps >= overlap_duration
                aligned_poses.append(transformed_poses[non_overlap_mask])
                aligned_timestamps.append(
                    current.timestamps[non_overlap_mask] +
                    prev_chunk_info["end_time"]
                )

            # Concatenate all poses
            all_poses = np.concatenate(aligned_poses, axis=0)
            all_timestamps = np.concatenate(aligned_timestamps, axis=0)

            return MergedResult(
                success=True,
                poses=all_poses,
                timestamps=all_timestamps,
                total_frames=len(all_poses),
                alignment_errors=alignment_errors,
            )

        except Exception as e:
            logger.error(f"Failed to merge chunks: {e}")
            return MergedResult(success=False, error=str(e))

    def _get_poses_in_timerange(
        self,
        poses: np.ndarray,
        timestamps: np.ndarray,
        start_time: float,
        end_time: float,
    ) -> np.ndarray:
        """Get poses within a time range."""
        mask = (timestamps >= start_time) & (timestamps <= end_time)
        return poses[mask]

    def _compute_alignment_transform(
        self,
        source_poses: np.ndarray,
        target_poses: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Compute rigid transform to align source poses to target poses.

        Uses Procrustes analysis on camera positions, optionally refined with ICP.

        Args:
            source_poses: Nx4x4 source camera poses
            target_poses: Mx4x4 target camera poses

        Returns:
            (4x4 transform matrix, alignment error in meters)
        """
        # Extract camera positions (translation component)
        source_positions = source_poses[:, :3, 3]
        target_positions = target_poses[:, :3, 3]

        # Subsample if too many points
        max_points = 100
        if len(source_positions) > max_points:
            indices = np.linspace(0, len(source_positions) - 1, max_points, dtype=int)
            source_positions = source_positions[indices]
        if len(target_positions) > max_points:
            indices = np.linspace(0, len(target_positions) - 1, max_points, dtype=int)
            target_positions = target_positions[indices]

        # Use Procrustes analysis for initial alignment
        transform, error = self._procrustes_alignment(
            source_positions, target_positions
        )

        # Optionally refine with ICP
        if self.use_icp:
            transform, error = self._icp_refinement(
                source_positions,
                target_positions,
                transform,
            )

        return transform, error

    def _procrustes_alignment(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Compute rigid alignment using Procrustes analysis.

        Args:
            source: Nx3 source points
            target: Mx3 target points

        Returns:
            (4x4 transform matrix, RMSE error)
        """
        # Center both point sets
        source_centroid = np.mean(source, axis=0)
        target_centroid = np.mean(target, axis=0)

        source_centered = source - source_centroid
        target_centered = target - target_centroid

        # Find nearest neighbors (simple approach)
        # For each source point, find closest target point
        min_len = min(len(source_centered), len(target_centered))
        source_centered = source_centered[:min_len]
        target_centered = target_centered[:min_len]

        # Compute rotation using SVD
        H = source_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = target_centroid - R @ source_centroid

        # Build 4x4 transform
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t

        # Compute error
        transformed = (R @ source.T).T + t
        min_len = min(len(transformed), len(target))
        error = np.sqrt(np.mean(np.sum((transformed[:min_len] - target[:min_len]) ** 2, axis=1)))

        return transform, error

    def _icp_refinement(
        self,
        source: np.ndarray,
        target: np.ndarray,
        initial_transform: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Refine alignment using ICP (Iterative Closest Point).

        Args:
            source: Nx3 source points
            target: Mx3 target points
            initial_transform: Initial 4x4 transform

        Returns:
            (Refined 4x4 transform, final error)
        """
        # Apply initial transform
        source_h = np.hstack([source, np.ones((len(source), 1))])
        current_source = (initial_transform @ source_h.T).T[:, :3]

        transform = initial_transform.copy()

        for iteration in range(self.icp_iterations):
            # Find nearest neighbors
            distances = np.zeros(len(current_source))
            correspondences = np.zeros(len(current_source), dtype=int)

            for i, p in enumerate(current_source):
                dists = np.sum((target - p) ** 2, axis=1)
                correspondences[i] = np.argmin(dists)
                distances[i] = np.sqrt(dists[correspondences[i]])

            # Reject outliers
            threshold = np.median(distances) * 2
            inliers = distances < threshold

            if np.sum(inliers) < 10:
                break

            # Compute transform for inliers
            source_inliers = current_source[inliers]
            target_inliers = target[correspondences[inliers]]

            delta_transform, _ = self._procrustes_alignment(source_inliers, target_inliers)

            # Apply delta
            transform = delta_transform @ transform
            current_source = (delta_transform @ np.hstack([current_source, np.ones((len(current_source), 1))]).T).T[:, :3]

            # Check convergence
            delta_t = np.linalg.norm(delta_transform[:3, 3])
            delta_r = np.arccos(np.clip((np.trace(delta_transform[:3, :3]) - 1) / 2, -1, 1))

            if delta_t < 0.001 and delta_r < 0.001:  # 1mm, 0.05 degrees
                break

        # Final error
        error = np.mean(distances[inliers]) if np.any(inliers) else float("inf")

        return transform, error

    def _apply_transform(
        self,
        poses: np.ndarray,
        transform: np.ndarray,
    ) -> np.ndarray:
        """Apply transform to all poses.

        Args:
            poses: Nx4x4 camera poses
            transform: 4x4 transform to apply

        Returns:
            Nx4x4 transformed poses
        """
        return np.array([transform @ pose for pose in poses])


class ParallelChunkProcessor:
    """Coordinate parallel processing of video chunks.

    This class manages:
    - Creating chunks from a long video
    - Dispatching chunk processing jobs to Cloud Tasks
    - Tracking chunk completion
    - Triggering merge when all chunks complete
    """

    def __init__(
        self,
        chunk_config: Optional[ChunkConfig] = None,
        queue_config: Optional[Dict[str, Any]] = None,
    ):
        self.chunker = VideoChunker(chunk_config)
        self.merger = ChunkMerger()
        self.queue_config = queue_config or {}

    def process_video(
        self,
        video_path: Path,
        session_id: str,
        output_dir: Path,
        gcs_bucket: str,
        parallel: bool = True,
    ) -> Dict[str, Any]:
        """Process a video, chunking if necessary.

        Args:
            video_path: Path to source video
            session_id: Session ID
            output_dir: Local output directory
            gcs_bucket: GCS bucket for uploads
            parallel: Whether to process chunks in parallel

        Returns:
            Processing result or job tracking info
        """
        # Check if chunking is needed
        if not self.chunker.should_chunk(video_path):
            logger.info("Video is short enough, processing without chunking")
            return {"chunked": False, "session_id": session_id}

        # Create chunks
        chunks_dir = output_dir / "chunks"
        chunks = self.chunker.create_chunks(video_path, chunks_dir, session_id)

        if not chunks:
            return {"error": "Failed to create chunks"}

        logger.info(f"Created {len(chunks)} chunks for parallel processing")

        if parallel:
            # Dispatch to Cloud Tasks for parallel processing
            return self._dispatch_parallel(chunks, session_id, gcs_bucket)
        else:
            # Process sequentially (for testing)
            return self._process_sequential(chunks, session_id, output_dir)

    def _dispatch_parallel(
        self,
        chunks: List[VideoChunk],
        session_id: str,
        gcs_bucket: str,
    ) -> Dict[str, Any]:
        """Dispatch chunks for parallel processing via Cloud Tasks.

        Args:
            chunks: List of VideoChunk objects
            session_id: Session ID
            gcs_bucket: GCS bucket

        Returns:
            Job tracking info
        """
        from .utils.cloud_tasks import get_task_queue, TaskPriority
        from .utils.gcs import GCSClient

        queue = get_task_queue()
        gcs = GCSClient()

        task_ids = []

        for chunk in chunks:
            # Upload chunk video to GCS
            chunk_gcs_path = f"sessions/{session_id}/chunks/chunk_{chunk.chunk_id:03d}.mp4"
            gcs.upload_file(
                str(chunk.video_path),
                gcs_bucket,
                chunk_gcs_path,
            )

            # Create chunk manifest
            chunk_manifest = {
                "session_id": session_id,
                "chunk_id": chunk.chunk_id,
                "video_uri": f"gs://{gcs_bucket}/{chunk_gcs_path}",
                "start_time": chunk.start_time,
                "end_time": chunk.end_time,
                "frame_offset": chunk.frame_offset,
                "is_chunk": True,
            }

            manifest_path = f"sessions/{session_id}/chunks/chunk_{chunk.chunk_id:03d}_manifest.json"
            gcs.upload_json(chunk_manifest, gcs_bucket, manifest_path)

            # Create task
            task_id = queue.create_pipeline_task(
                capture_id=f"{session_id}_chunk_{chunk.chunk_id:03d}",
                session_manifest_uri=f"gs://{gcs_bucket}/{manifest_path}",
                output_base=f"gs://{gcs_bucket}/sessions/{session_id}/chunks/chunk_{chunk.chunk_id:03d}",
                priority=TaskPriority.NORMAL,
            )

            if task_id:
                task_ids.append(task_id)

        return {
            "chunked": True,
            "session_id": session_id,
            "total_chunks": len(chunks),
            "task_ids": task_ids,
            "status": "dispatched",
        }

    def _process_sequential(
        self,
        chunks: List[VideoChunk],
        session_id: str,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Process chunks sequentially (for testing).

        Args:
            chunks: List of VideoChunk objects
            session_id: Session ID
            output_dir: Output directory

        Returns:
            Processing result
        """
        # This would call the actual pipeline for each chunk
        # For now, return placeholder
        return {
            "chunked": True,
            "session_id": session_id,
            "total_chunks": len(chunks),
            "status": "sequential_processing",
            "note": "Sequential processing not yet implemented",
        }

    def on_chunk_complete(
        self,
        session_id: str,
        chunk_id: int,
        result: ChunkResult,
        gcs_bucket: str,
    ) -> Optional[MergedResult]:
        """Handle completion of a chunk, potentially triggering merge.

        Args:
            session_id: Session ID
            chunk_id: Completed chunk ID
            result: Chunk processing result
            gcs_bucket: GCS bucket

        Returns:
            MergedResult if all chunks complete, None otherwise
        """
        from .utils.firestore import get_job_tracker

        tracker = get_job_tracker()

        # Save chunk result to Firestore
        # ... (implementation depends on Firestore schema)

        # Check if all chunks complete
        # ... (query Firestore for all chunk statuses)

        # If all complete, trigger merge
        # ... (load all results and merge)

        return None  # Placeholder


# =============================================================================
# Utility Functions
# =============================================================================

def chunk_video_for_parallel_processing(
    video_uri: str,
    output_base: str,
    chunk_duration: int = 60,
    overlap: int = 10,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Split long video into overlapping chunks for parallel SLAM.

    High-level utility function for chunking videos.

    Args:
        video_uri: GCS URI to video file
        output_base: GCS base path for outputs
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap between chunks in seconds
        session_id: Optional session ID

    Returns:
        Dictionary with chunk info and task IDs
    """
    import tempfile
    from pathlib import Path
    from .utils.gcs import GCSClient, GCSPath

    gcs = GCSClient()
    parsed = GCSPath.from_uri(video_uri)

    # Generate session ID if not provided
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())[:8]

    # Download video to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        local_video = Path(tmpdir) / "video.mp4"
        gcs.download_file(parsed.bucket, parsed.blob, str(local_video))

        # Create chunker with config
        config = ChunkConfig(
            chunk_duration=float(chunk_duration),
            overlap_duration=float(overlap),
        )
        processor = ParallelChunkProcessor(config)

        # Process video
        output_parsed = GCSPath.from_uri(output_base)
        result = processor.process_video(
            video_path=local_video,
            session_id=session_id,
            output_dir=Path(tmpdir) / "output",
            gcs_bucket=output_parsed.bucket,
            parallel=True,
        )

        return result
