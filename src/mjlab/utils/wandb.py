import shutil
from pathlib import Path

import wandb
from tqdm import tqdm


def download_motions_from_wandb(
  wandb_entity: str,
  wandb_project: str,
  cache_dir: Path | str | None = None,
  artifact_type: str = "motions",
) -> Path:
  """Download motion artifacts from wandb registry.

  Downloads all artifacts of the specified type from all runs in the wandb project
  and organizes them in a directory structure compatible with MultiMotionLoader.

  Args:
      wandb_entity: Wandb entity/username (e.g., "ATARITUM")
      wandb_project: Wandb project name (e.g., "sbto_v1")
      cache_dir: Directory to cache downloaded artifacts. If None, uses ~/.cache/mjlab/wandb_motions
      artifact_type: Type of artifact to download (default: "motions")

  Returns:
      Path to the directory containing downloaded motions organized as:
      cache_dir/{project}/motions/{run_name}/{artifact_name}/motion.npz
  """
  # Set up cache directory
  if cache_dir is None:
    cache_dir = Path.home() / ".cache" / "mjlab" / "wandb_motions"
  else:
    cache_dir = Path(cache_dir)

  project_cache_dir = cache_dir / wandb_project
  motions_dir = project_cache_dir / "motions"
  motions_dir.mkdir(parents=True, exist_ok=True)

  # Check what's already in cache
  cached_motions = {}
  if motions_dir.exists():
    for motion_file in motions_dir.rglob("motion.npz"):
      # Get relative path from motions_dir to determine artifact name
      rel_path = motion_file.relative_to(motions_dir)
      # For registry artifacts: artifact_name/motion.npz
      # For run artifacts: run_name/artifact_name/motion.npz
      if len(rel_path.parts) == 2:
        artifact_name = rel_path.parts[0]
        cached_motions[artifact_name] = motion_file
      elif len(rel_path.parts) == 3:
        # Run-based artifacts: run_name/artifact_name/motion.npz
        run_name, artifact_name = rel_path.parts[0], rel_path.parts[1]
        key = f"{run_name}/{artifact_name}"
        cached_motions[key] = motion_file

  # Initialize wandb API
  if wandb is None:
    raise ImportError("wandb is required but not available")
  api = wandb.Api()

  downloaded_count = 0
  skipped_count = 0

  # Try to get artifacts from registry first
  try:
    registry_artifacts = api.artifacts(  # type: ignore
      f"{wandb_entity}/{wandb_project}",
      type=artifact_type,  # type: ignore
    )
    artifact_list = list(registry_artifacts)

    if not artifact_list:
      print(f"No artifacts found in {wandb_entity}/{wandb_project}")
      return motions_dir

    # Download artifacts from registry
    for artifact in tqdm(artifact_list, desc="Downloading artifacts", unit="artifact"):
      artifact_name = artifact.name

      # Use artifact name as the trajectory name (since it's the collection name)
      # Create directory: {artifact_name}
      artifact_dir = motions_dir / artifact_name
      motion_file = artifact_dir / "motion.npz"

      # Skip if already in cache
      if artifact_name in cached_motions or motion_file.exists():
        skipped_count += 1
        continue

      try:
        # Download artifact to temporary directory
        artifact_dir.mkdir(parents=True, exist_ok=True)
        temp_download_dir = artifact_dir / "temp"
        temp_download_dir.mkdir(exist_ok=True)

        artifact_path = artifact.download(root=str(temp_download_dir))

        # Check if motion.npz is in the artifact root or a subdirectory
        artifact_path_obj = Path(artifact_path)
        motion_file_candidate = artifact_path_obj / "motion.npz"

        found_motion = False
        if motion_file_candidate.exists():
          # Move to expected location
          shutil.move(str(motion_file_candidate), str(motion_file))
          found_motion = True
        else:
          # Look for motion.npz in subdirectories
          for motion_candidate in artifact_path_obj.rglob("motion.npz"):
            # Move to expected location
            shutil.move(str(motion_candidate), str(motion_file))
            found_motion = True
            break

        if not found_motion:
          # Clean up empty directory
          if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
          continue

        # Clean up temporary download directory
        if temp_download_dir.exists():
          shutil.rmtree(temp_download_dir)

        downloaded_count += 1

      except Exception:
        # Clean up on error
        if artifact_dir.exists():
          shutil.rmtree(artifact_dir)
        continue

  except Exception as e:
    print(f"Warning: Could not access registry: {e}")

  print(
    f"\nDownloaded {downloaded_count} new artifacts, {skipped_count} already cached"
  )

  # Verify we have at least one motion file
  motion_files = list(motions_dir.rglob("motion.npz"))
  print(f"Total number of motions: {len(motion_files)}")
  if not motion_files:
    raise RuntimeError(
      f"No motion.npz files found in {motions_dir}. "
      f"Please check that artifacts are properly uploaded to wandb."
    )

  return motions_dir
