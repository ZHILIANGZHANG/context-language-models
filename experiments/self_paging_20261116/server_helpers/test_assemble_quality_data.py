from assemble_quality_data import thin_plain_next

def test_loss_only_and_deterministic():
 rec=dict(meta={'seed':2000},tokens=list(range(100)),page_of=[0]*100,seg_of=[0]*100,loss_mask=[1]*100,ctrl_mask=[1]*100)
 turns=[dict(start=i,length=10,plain_next=i<80) for i in range(0,100,10)]
 a=thin_plain_next(rec,turns);b=thin_plain_next(rec,turns)
 assert a==b and rec['loss_mask']==[1]*100
 assert a['loss_mask'][80:]==[1]*20 and sum(a['loss_mask'])<100
 for k in ['tokens','page_of','seg_of','ctrl_mask']:assert a[k]==rec[k]
 assert set(a['loss_mask'])<={0,1}
