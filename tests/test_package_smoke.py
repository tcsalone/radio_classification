"""Smoke test: every package imports without optional heavy deps."""

from __future__ import annotations


def test_top_level_imports() -> None:
    import radio_classifier
    from radio_classifier import (
        acoustic,
        cli,
        commercials,
        fingerprint,
        ingest,
        music,
        persistence,
        pipeline,
        reports,
        segments,
        speech,
    )

    assert radio_classifier.__version__ == "0.1.0"
    # The packages export what their __init__ promised.
    assert hasattr(segments, "BroadcastCategory")
    assert hasattr(persistence, "BroadcastStore")
    assert hasattr(fingerprint, "AudfprintIndex")
    assert hasattr(acoustic, "AcousticLabel")
    assert hasattr(speech, "OllamaSpeechClassifier")
    assert hasattr(commercials, "CommercialIdentityResolver")
    assert hasattr(pipeline, "FunnelOrchestrator")
    assert hasattr(reports, "format_commercials")
    assert hasattr(music, "identify_window_sync")
    assert hasattr(ingest, "AudioWindow")
    assert hasattr(cli, "main")


def test_optional_deps_not_eagerly_imported() -> None:
    """Importing radio_classifier must NOT eagerly load the heavy optional deps.

    The runtime ingest path uses lazy imports so that an operator without
    ``[acoustic]`` / ``[fingerprint]`` / ``[shazam]`` installed can still run
    CLI subcommands like ``report`` or ``db init``.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import sys
        import radio_classifier
        import radio_classifier.cli
        import radio_classifier.acoustic
        import radio_classifier.fingerprint
        import radio_classifier.music
        import radio_classifier.speech
        import radio_classifier.seeding
        forbidden_eager = [
            "tensorflow_hub", "tensorflow",
            "faster_whisper",
            "shazamio",
            "bs4", "requests",
            "yt_dlp",
        ]
        offenders = [name for name in forbidden_eager if name in sys.modules]
        if offenders:
            print("EAGER:" + ",".join(offenders))
            sys.exit(1)
        print("CLEAN")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "CLEAN" in proc.stdout
