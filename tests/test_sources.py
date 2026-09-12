from unittest.mock import patch, MagicMock

import pytest

requests = pytest.importorskip("requests")

from terra_texture.sources import (  # noqa: E402
    stac_search,
    mosaic_dem_urls,
    arcticdem_mosaic_urls,
    rema_mosaic_urls,
    _item_asset_href,
    _bbox_intersects,
    list_stac_collections,
    stac_collection_items,
    opentopography_dem_urls,
)


def _fake_response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _item(item_id, dem_href):
    return {
        "type": "Feature",
        "id": item_id,
        "assets": {"dem": {"href": dem_href}},
    }


def test_stac_search_single_page():
    page = {
        "features": [_item("tile_a", "https://example.com/a_dem.tif")],
        "links": [],
    }
    with patch("requests.post", return_value=_fake_response(page)) as mock_post:
        items = stac_search(["some-collection"], (0, 0, 1, 1))

    assert len(items) == 1
    assert items[0]["id"] == "tile_a"
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["collections"] == ["some-collection"]
    assert call_kwargs["json"]["bbox"] == [0, 0, 1, 1]


def test_stac_search_follows_pagination():
    page1 = {
        "features": [_item("tile_a", "https://example.com/a_dem.tif")],
        "links": [{"rel": "next", "href": "https://example.com/search?page=2", "method": "GET"}],
    }
    page2 = {
        "features": [_item("tile_b", "https://example.com/b_dem.tif")],
        "links": [],
    }
    with patch("requests.post", return_value=_fake_response(page1)), \
         patch("requests.get", return_value=_fake_response(page2)) as mock_get:
        items = stac_search(["some-collection"], (0, 0, 1, 1))

    assert [item["id"] for item in items] == ["tile_a", "tile_b"]
    mock_get.assert_called_once()


def test_stac_search_respects_max_items():
    page = {
        "features": [
            _item("tile_a", "https://example.com/a_dem.tif"),
            _item("tile_b", "https://example.com/b_dem.tif"),
        ],
        "links": [{"rel": "next", "href": "https://example.com/search?page=2", "method": "GET"}],
    }
    with patch("requests.post", return_value=_fake_response(page)):
        items = stac_search(["some-collection"], (0, 0, 1, 1), max_items=1)

    assert len(items) == 1


def test_item_asset_href_missing_key_lists_available():
    item = _item("tile_a", "https://example.com/a_dem.tif")
    with pytest.raises(KeyError, match="hillshade"):
        _item_asset_href(item, "hillshade")


def test_mosaic_dem_urls_rejects_bad_resolution():
    with pytest.raises(ValueError, match="resolution"):
        mosaic_dem_urls("arcticdem", (0, 0, 1, 1), resolution=7)


def test_arcticdem_mosaic_urls_builds_correct_collection():
    page = {"features": [_item("58_10_2m_v4.1", "https://example.com/dem.tif")], "links": []}
    with patch("requests.post", return_value=_fake_response(page)) as mock_post:
        urls = arcticdem_mosaic_urls((0, 0, 1, 1), resolution=2)

    assert urls == ["https://example.com/dem.tif"]
    body = mock_post.call_args.kwargs["json"]
    assert body["collections"] == ["arcticdem-mosaics-v4.1-2m"]


def test_rema_mosaic_urls_builds_correct_collection():
    page = {"features": [_item("44_06_32m_v2.0", "https://example.com/dem.tif")], "links": []}
    with patch("requests.post", return_value=_fake_response(page)) as mock_post:
        urls = rema_mosaic_urls((0, 0, 1, 1), resolution=32)

    assert urls == ["https://example.com/dem.tif"]
    body = mock_post.call_args.kwargs["json"]
    assert body["collections"] == ["rema-mosaics-v2.0-32m"]


# -- static-catalog engine (OpenTopography-style: no /search endpoint) --

def _item_with_bbox(item_id, bbox, dem_href):
    return {
        "type": "Feature",
        "id": item_id,
        "bbox": list(bbox),
        "assets": {"data": {"href": dem_href}},
    }


def test_bbox_intersects():
    assert _bbox_intersects((0, 0, 2, 2), (1, 1, 3, 3))
    assert not _bbox_intersects((0, 0, 1, 1), (2, 2, 3, 3))


def test_list_stac_collections_follows_child_links():
    catalog = {
        "type": "Catalog",
        "links": [
            {"rel": "self", "href": "https://example.com/catalog.json"},
            {"rel": "child", "href": "./srtm/collection.json", "title": "SRTM GL1"},
            {"rel": "child", "href": "https://example.com/cop30/collection.json", "title": "COP30"},
        ],
    }
    with patch("requests.get", return_value=_fake_response(catalog)):
        collections = list_stac_collections("https://example.com/catalog.json")

    ids = [c["id"] for c in collections]
    assert ids == ["SRTM GL1", "COP30"]
    # relative href resolved against the catalog URL
    assert collections[0]["href"] == "https://example.com/srtm/collection.json"
    assert collections[1]["href"] == "https://example.com/cop30/collection.json"


def test_stac_collection_items_filters_by_bbox_client_side():
    collection_doc = {
        "links": [{"rel": "items", "href": "https://example.com/srtm/items"}],
    }
    items_page = {
        "features": [
            _item_with_bbox("inside", (0, 0, 1, 1), "https://example.com/inside_dem.tif"),
            _item_with_bbox("outside", (10, 10, 11, 11), "https://example.com/outside_dem.tif"),
        ],
        "links": [],
    }

    def fake_get(url, params=None, timeout=None):
        if url == "https://example.com/srtm/collection.json":
            return _fake_response(collection_doc)
        return _fake_response(items_page)

    with patch("requests.get", side_effect=fake_get):
        items = stac_collection_items(
            {"href": "https://example.com/srtm/collection.json"}, bbox=(0, 0, 2, 2),
        )

    assert [item["id"] for item in items] == ["inside"]


def test_stac_collection_items_paginates():
    collection_doc = {"links": [{"rel": "items", "href": "https://example.com/items"}]}
    page1 = {
        "features": [_item_with_bbox("a", (0, 0, 1, 1), "https://example.com/a.tif")],
        "links": [{"rel": "next", "href": "https://example.com/items?page=2"}],
    }
    page2 = {
        "features": [_item_with_bbox("b", (0, 0, 1, 1), "https://example.com/b.tif")],
        "links": [],
    }
    call_urls = []

    def fake_get(url, params=None, timeout=None):
        call_urls.append(url)
        if url == "https://example.com/collection.json":
            return _fake_response(collection_doc)
        if url == "https://example.com/items":
            return _fake_response(page1)
        return _fake_response(page2)

    with patch("requests.get", side_effect=fake_get):
        items = stac_collection_items(
            {"href": "https://example.com/collection.json"}, bbox=(0, 0, 2, 2),
        )

    assert [item["id"] for item in items] == ["a", "b"]


def test_stac_collection_items_no_items_link_raises():
    with pytest.raises(ValueError, match="items"):
        stac_collection_items({"href": "https://example.com/x", "links": []})


def test_opentopography_dem_urls_resolves_collection_by_id():
    catalog = {
        "links": [{"rel": "child", "href": "https://example.com/srtm/collection.json", "title": "SRTM GL1"}],
    }
    collection_doc = {"links": [{"rel": "items", "href": "https://example.com/srtm/items"}]}
    items_page = {
        "features": [_item_with_bbox("tile1", (0, 0, 1, 1), "https://example.com/tile1_dem.tif")],
        "links": [],
    }

    def fake_get(url, params=None, timeout=None):
        if url == "https://example.com/catalog.json":
            return _fake_response(catalog)
        if url == "https://example.com/srtm/collection.json":
            return _fake_response(collection_doc)
        return _fake_response(items_page)

    with patch("requests.get", side_effect=fake_get):
        urls = opentopography_dem_urls(
            "SRTM GL1", (0, 0, 2, 2), catalog_url="https://example.com/catalog.json",
        )

    assert urls == ["https://example.com/tile1_dem.tif"]


def test_opentopography_dem_urls_unknown_collection_lists_available():
    catalog = {
        "links": [{"rel": "child", "href": "https://example.com/srtm/collection.json", "title": "SRTM GL1"}],
    }
    with patch("requests.get", return_value=_fake_response(catalog)):
        with pytest.raises(ValueError, match="SRTM GL1"):
            opentopography_dem_urls(
                "NoSuchDataset", (0, 0, 1, 1), catalog_url="https://example.com/catalog.json",
            )
