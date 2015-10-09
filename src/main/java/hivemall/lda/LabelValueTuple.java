package hivemall.lda;

public class LabelValueTuple {
	private final String _label;
	private final float _value;
	
	public LabelValueTuple(String label, float value){
		_label = label;
		_value = value;
	}
	
	public String getLabel(){
		return _label;
	}

	public float getValue(){
		return _value;
	}
}
