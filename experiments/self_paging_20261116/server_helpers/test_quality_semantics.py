from quality_semantics import semantic_errors

def test_supported_and_contradictory_claims():
 state=({1,2},{4});calls=[{'name':'store','arguments':{'shelf':3}}]
 for text in ['I will store I-5 on S3 as it is the lowest free shelf.','Lowest free shelf: S3.','S3 is the lowest-numbered shelf that is empty.']:
  assert not semantic_errors(text,calls,state),text
 for text in ['I will store I-5 on S1.','S2 is the lowest free shelf.','Lowest free shelf: S2.']:
  assert semantic_errors(text,calls,state),text
 assert not semantic_errors('S1 occupied, S2 occupied, S3 empty. Lowest free shelf: S3.',calls,state)
 assert semantic_errors('I will store I-5 on S3.',[{'name':'kv_out','arguments':{'pages':'3'}}],state)
