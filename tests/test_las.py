from examples.synthetic_demo import write_synthetic_las
from coal_boundary.las import iter_las_points, read_las_header


def test_synthetic_las_header_and_chunk(tmp_path):
    path = tmp_path / 'synthetic.las'
    write_synthetic_las(path)
    header = read_las_header(path)
    assert header.point_format == 3
    assert header.point_record_length == 34
    assert header.point_count > 1000
    chunk = next(iter_las_points(header, 10))
    assert len(chunk['x']) == 10
    assert chunk['red'].max() <= 255
