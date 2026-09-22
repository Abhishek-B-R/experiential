"""Atomic publication of a verified staged project into the shared database."""

from __future__ import annotations

import os

from exp.common.project.database import content_database_path, project_connection
from exp.common.project.paths import ProjectPaths

_TABLES = (
    "project_config_versions",
    "project_config_heads",
    "project_artifacts",
    "project_artifact_inputs",
    "project_artifact_files",
    "project_state_records",
    "project_state_events",
)


def publish_restored_project(staged: ProjectPaths, destination: ProjectPaths) -> None:
    """Publish staged files before atomically installing the verified project rows.

    The source database is read-only. A single destination commit installs every
    project row; existing capture, trace and other project rows remain untouched.

    Args:
        staged: Independently verified staging root and project identity.
        destination: Shared destination root and the same project identity.
    """
    if staged.project_id != destination.project_id:
        raise ValueError("restored project identity changed before publication")
    published = False
    try:
        with project_connection(destination.root, write=True) as connection:
            connection.execute(
                "ATTACH DATABASE ? AS restored",
                (f"{content_database_path(staged.root).as_uri()}?mode=ro",),
            )
            for table in _TABLES:
                if connection.execute(
                    f"SELECT 1 FROM main.{table} WHERE project_id=? LIMIT 1",
                    (destination.project_id,),
                ).fetchone():
                    raise ValueError("restore destination already contains project state")
            os.rename(staged.project_directory, destination.project_directory)
            published = True
            descriptor = os.open(destination.projects_directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            for table in _TABLES:
                connection.execute(
                    f"INSERT INTO main.{table} SELECT * FROM restored.{table} WHERE project_id=?",
                    (destination.project_id,),
                )
    except BaseException:
        if published:
            os.rename(destination.project_directory, staged.project_directory)
        raise
