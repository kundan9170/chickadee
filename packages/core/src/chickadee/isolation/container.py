"""docker/podman wrapper — opt-in, strongest, and adds a prerequisite."""

from __future__ import annotations

import shutil

from chickadee.isolation.base import first_line, has, probe
from chickadee.isolation.types import Isolation, SandboxLevel, mask_for
from chickadee.process import Completed

IMAGE = "chickadee-sandbox:latest"


class Container:
    def __init__(self, image: str = IMAGE, enabled: bool = False) -> None:
        self.image = image
        # Opt-in by default: a container backend that engaged automatically would change
        # the toolchain a run sees without anyone asking for it.
        self.enabled = enabled

    def _runtime(self) -> str | None:
        for candidate in ("podman", "docker"):
            if has(candidate):
                return candidate
        return None

    def available(self) -> bool:
        """Enabled, and it runs, not that a runtime is on the path."""
        return probe(self).ran

    # -- what the probe asks -------------------------------------------------------

    def probe_key(self) -> tuple[str, ...]:
        runtime = self._runtime()
        found = shutil.which(runtime) if runtime is not None else None
        return (str(self.enabled), found or "", self.image)

    def not_ready(self) -> tuple[str, str] | None:
        if not self.enabled:
            return (
                "the container backend is opt-in and not enabled",
                'set sandbox.backend = "container" in the engine config to use it',
            )
        if self._runtime() is None:
            return (
                "neither podman nor docker is on the PATH",
                'install podman or docker, or leave sandbox.backend at "native"',
            )
        return None

    def diagnose(self, done: Completed) -> tuple[str, str]:
        runtime = self._runtime() or "docker"
        said = first_line(done)
        lowered = said.lower()
        if "daemon" in lowered or "cannot connect" in lowered:
            return (
                f"the {runtime} daemon is not running: {said}",
                f"start it (`systemctl start {runtime}`, or the desktop app)",
            )
        if "unable to find image" in lowered or "image not known" in lowered:
            return (
                f"the image {self.image} is not present: {said}",
                f"build or pull {self.image} with {runtime}",
            )
        return f"{runtime} exited {done.exit_code}: {said}", f"run `{runtime} info` to see why"

    def level(self) -> SandboxLevel:
        return SandboxLevel.CONTAINER

    def wrap(self, isolation: Isolation) -> tuple[str, ...]:
        runtime = self._runtime() or "docker"
        mount = "rw" if isolation.repo_writable else "ro"
        argv = [
            runtime,
            "run",
            "--rm",
            "-w",
            str(isolation.cwd),
            "-v",
            f"{isolation.cwd}:{isolation.cwd}:{mount}",
        ]
        if isolation.repo_writable:
            for extra in isolation.writable:
                argv += ["-v", f"{extra}:{extra}:rw"]
            # After the writable mounts, so each is mounted over the one it lies under.
            for kept in isolation.read_only:
                argv += ["-v", f"{kept}:{kept}:ro"]
        # The harness's scratch, writable whatever the wrap.
        for extra in (*isolation.scratch, *((isolation.tmp,) if isolation.tmp else ())):
            argv += ["-v", f"{extra}:{extra}:rw"]
        # The same mode-000 scratch entries, as read-only volumes over each
        # denied path. Root-bounded like bubblewrap; the mount is what fails the read.
        for denied in isolation.deny_read:
            argv += ["-v", f"{mask_for(denied)}:{denied}:ro"]
        if isolation.network_off:
            argv += ["--network", "none"]
        argv.append(self.image)
        return (*argv, *isolation.argv)
