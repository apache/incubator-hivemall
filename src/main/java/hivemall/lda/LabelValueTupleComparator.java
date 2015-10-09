package hivemall.lda;

import java.util.Comparator;

public class LabelValueTupleComparator implements Comparator<LabelValueTuple>{
	
	@Override
	public int compare(LabelValueTuple o1, LabelValueTuple o2) {
		float v1 = o1.getValue();
		float v2 = o2.getValue();
		
		if(v1 < v2){
			return 1;
		}else if(v2 < v1){
			return -1;
		}else{
			return 0;
		}
	}
}
