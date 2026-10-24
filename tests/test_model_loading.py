import sys

from app.services import model_loader


def test_windows_loss_alias_registered_before_loading(monkeypatch, tmp_path):
    from sklearn._loss import _loss
    path = tmp_path/'model.joblib'
    path.write_bytes(b'trusted-test-artifact')
    monkeypatch.delitem(sys.modules, '_loss', raising=False)
    def load(filename):
        assert sys.modules['_loss'] is _loss
        return {'model': object(), 'feature_columns': ['foF2'], 'metadata': {}}
    monkeypatch.setattr(model_loader.joblib, 'load', load)
    model_loader.load_production_model.cache_clear()
    try:
        assert model_loader.load_production_model(str(path))['feature_columns'] == ['foF2']
    finally:
        model_loader.load_production_model.cache_clear()
