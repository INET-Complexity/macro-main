from macro_data.configuration.region import Region


def test_from_code():
    region = Region.from_code("CAN_ON", "Ontario")

    assert region == "CAN_ON"

    assert region.name == "Ontario"

    assert region.parent_country == "CAN"

    assert region.parent_country.name == "CANADA"
