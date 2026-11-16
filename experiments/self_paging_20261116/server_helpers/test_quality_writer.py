from quality_writer import factual_errors

def test_claims():
 state=({1,2,4},{3})
 assert not factual_errors('S1 occupied, S3 quarantined, S5 empty.',state)
 assert factual_errors('S5 is occupied.',state)
 assert factual_errors('S1 through S4 are occupied.',state)
 assert factual_errors('S3 now occupied.',state)
 assert not factual_errors('S3 empty and quarantined.',state) # These properties are independent.
