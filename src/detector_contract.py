


def check_detector(settings, stride):
    assert [w * stride for w in settings['windows']] == [12, 16, 20], \
        'Input-event windows are not matched'
    assert settings['sigma'] * stride == 2, 'Input-event smoothing is not matched'
    assert settings['width'] * stride == 2, 'Input-event peak width is not matched'
    assert settings['distance'] * stride == 50
    assert settings['prominence'] == .30
