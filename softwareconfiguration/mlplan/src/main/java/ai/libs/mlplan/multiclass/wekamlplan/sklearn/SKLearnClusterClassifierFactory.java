package ai.libs.mlplan.multiclass.wekamlplan.sklearn;

import ai.libs.hasco.exceptions.ComponentInstantiationFailedException;
import ai.libs.hasco.model.ComponentInstance;
import ai.libs.jaicore.ml.scikitwrapper.ScikitLearnWrapper;
import ai.libs.jaicore.ml.scikitwrapper.ScikitLearnWrapper.ProblemType;
import weka.classifiers.Classifier;

public class SKLearnClusterClassifierFactory extends SKLearnClassifierFactory {

	@Override
	public Classifier getComponentInstantiation(final ComponentInstance groundComponent) throws ComponentInstantiationFailedException {
		Classifier classifier = super.getComponentInstantiation(groundComponent);
		if(classifier == null){
			return null;
		}else{
			((ScikitLearnWrapper)classifier).setProblemType(ProblemType.CLUSTERING);
			return classifier;
		}
	}
}
