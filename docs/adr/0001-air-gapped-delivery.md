# Air-gapped delivery: Mirror for dependencies, Binary drop for the rest

The system runs on a network with no internet access. That network has a **Mirror** of public package sources (Python packages, npm, GitHub repositories), populated on request. We install from the Mirror wherever it can serve a file, and carry only the remainder across by hand as a **Binary drop**.

- Python dependencies are pinned in `requirements.lock`, which doubles as the explicit list to request from the Mirror. The Installer runs pip against the Mirror; nothing downloads wheels by hand.
- The Binary drop carries what the Mirror cannot provide: the Qdrant server binary and, while the source repository is private, an archive of this system's own code. On the container route it carries saved system and Qdrant images instead of the Qdrant binary. Each file has a version and checksum.
- The repository records the Qdrant version and checksum it expects. The Installer (and the container route) refuse a drop that does not match, before changing anything.
- Moving our own code delivery from the hand-carried archive to the Mirror is deferred until the repository is public.

## Considered options

- **Carry everything by hand** (the earlier hand-run download and offline-install scripts): rejected. The Mirror already exists, and hand-carried wheels are an unreviewed, unpinned second path for the same packages.
- **Carry nothing, ask the Mirror for Qdrant too**: not possible today, since Qdrant releases are not available from it.

## Consequences

- Upgrading is the same command every time: build a drop on a connected machine, request any new pins from the Mirror, run the Installer.
- A wrong or corrupt drop is caught by the checksum check, not by a service failing to start.
