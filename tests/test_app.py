"""使用 Streamlit 官方 AppTest 验证页面交互；不需要操作真实数据库。"""
from streamlit.testing.v1 import AppTest
from ev_core import ROOT


def app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert "演示数据" not in [option for radio in at.sidebar.radio for option in radio.options]
    assert not at.exception
    return at


def test_dashboard_filter_and_empty_state():
    at = app()
    assert len(at.metric) == 4
    at.multiselect[0].set_value([]).run()
    assert not at.exception
    assert any("没有充电记录" in x.value for x in at.info)


def test_analytics_pages():
    at = app()
    at.sidebar.radio[0].set_value("可视化分析").run()
    assert not at.exception
    assert len(at.tabs) == 4
    at.selectbox[0].set_value("AC").run()
    assert not at.exception


def test_live_prediction():
    at = app()
    at.sidebar.radio[0].set_value("实时预测").run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert len(at.metric) == 1
    assert len(at.dataframe) == 1
