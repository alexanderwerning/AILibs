package ai.libs.mlplan.multiclass.wekamlplan.sklearn;

import ai.libs.mlplan.core.AbstractMLPlanBuilder;
import ai.libs.mlplan.multiclass.wekamlplan.MLPlanWekaClassifier;
import java.io.IOException;


public class SKLearnClusterMLPlanWekaClassifier extends MLPlanWekaClassifier {

	public SKLearnClusterMLPlanWekaClassifier(final AbstractMLPlanBuilder builder) throws IOException {
		super(builder);
	}

	public SKLearnClusterMLPlanWekaClassifier() throws IOException {
		super(AbstractMLPlanBuilder.forSKLearnCluster());
	}

}
