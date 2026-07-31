import json
import types

from trellis2 import pipelines


def test_from_pretrained_resolves_a_lazy_pipeline_class(tmp_path, monkeypatch):
    class LazyPipeline:
        @classmethod
        def from_pretrained(cls, path, **hub_kwargs):
            return path, hub_kwargs

    imported = []

    def import_module(name, package):
        imported.append((name, package))
        return types.SimpleNamespace(LazyPipeline=LazyPipeline)

    monkeypatch.setattr(pipelines, "__attributes", {"LazyPipeline": "lazy_pipeline"})
    monkeypatch.setattr(pipelines.importlib, "import_module", import_module)
    (tmp_path / "pipeline.json").write_text(json.dumps({"name": "LazyPipeline"}))

    result = pipelines.from_pretrained(str(tmp_path), local_files_only=True)

    assert result == (str(tmp_path), {"local_files_only": True})
    assert imported == [(".lazy_pipeline", "trellis2.pipelines")]
