"""Temporary synthetic workbook, with no field records."""
from openpyxl import Workbook
from coal_boundary.reference import load_reference_workbook


def test_generated_workbook_is_parsed(tmp_path):
    workbook = Workbook()
    ledger = workbook.active
    ledger.title = '测量台账'
    ledger.append(['id', 'name', '', '', '', 'base', '', '', 'volume'])
    ledger.append([1, 'synthetic-pile', None, None, None, 0, None, None, 250])
    details = workbook.create_sheet('顶点明细')
    details.append(['id', 'name', 'order', 'longitude', 'latitude'])
    for i, (lon, lat) in enumerate([(120, 30), (120.001, 30),
                                   (120.001, 30.001), (120, 30.001)]):
        details.append([1, 'synthetic-pile', i, lon, lat])
    path = tmp_path / 'synthetic-reference.xlsx'
    workbook.save(path)
    piles = load_reference_workbook(path)
    assert len(piles) == 1
    assert piles[0].true_volume_m3 == 250
    assert piles[0].polygon_projected.is_valid
