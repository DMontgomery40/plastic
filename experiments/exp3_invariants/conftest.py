def pytest_configure(config):
    config.addinivalue_line("markers", "documented_limitation: asserts a documented limitation is present")
    config.addinivalue_line("markers", "real_model: uses the trained lm_small_cpu checkpoint")
