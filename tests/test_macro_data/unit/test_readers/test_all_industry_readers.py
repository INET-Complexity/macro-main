from macro_data.readers import ALL_INDUSTRIES


def test__icio_industries(all_industries_readers):
    icio = all_industries_readers.icio[2014]
    assert set(ALL_INDUSTRIES).issubset(set(icio.industries))
