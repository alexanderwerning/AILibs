package ai.libs.mlplan;

import ai.libs.hasco.model.Component;
import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.model.ComponentUtil;
import ai.libs.hasco.model.NumericParameterDomain;
import ai.libs.hasco.model.Parameter;
import ai.libs.hasco.serialization.ComponentLoader;
import ai.libs.jaicore.basic.SQLAdapter;
import ai.libs.jaicore.basic.TimeOut;
import ai.libs.jaicore.basic.sets.PartialOrderedSet;
import ai.libs.jaicore.experiments.Experiment;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.ExperimentRunner;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterSQLHandle;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.experiments.exceptions.IllegalExperimentSetupException;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.DaviesBouldinMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.DensityBasedMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.DunnMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.IInternalClusteringValidationMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.SilhouetteMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.SquaredErrorMeasure;
import ai.libs.jaicore.ml.openml.OpenMLHelper;
import ai.libs.mlplan.ClusteringEvaluation.ClusteringResult;
import ai.libs.mlplan.core.AbstractMLPlanBuilder;
import ai.libs.mlplan.core.MLPlanSKLearnClusterBuilder;
import ai.libs.mlplan.core.events.ClassifierFoundEvent;
import ai.libs.mlplan.multiclass.MLPlanClassifierConfig;
import ai.libs.mlplan.multiclass.wekamlplan.sklearn.SKLearnClassifierFactory;
import ai.libs.mlplan.multiclass.wekamlplan.sklearn.SKLearnClusterClassifierFactory;
import ai.libs.mlplan.multiclass.wekamlplan.sklearn.SKLearnClusterMLPlanWekaClassifier;
import com.google.common.eventbus.Subscribe;
import java.io.File;
import java.sql.SQLException;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import org.aeonbits.owner.ConfigFactory;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Remove;


public class MlplanClusterSearch implements IExperimentSetEvaluator {

	private static final IAutoMLForClusteringExperimentConfig CONFIG = ConfigFactory.create(IAutoMLForClusteringExperimentConfig.class);
	private static final AtomicInteger status = new AtomicInteger(1);
	private static final String FS_SEARCH_SPACE_CONFIG = "softwareconfiguration/mlplan/resources/automl/searchmodels/sklearn/sklearn-cluster-mlplan.json";
	private static final String DEF_REQUESTED_HASCO_INTERFACE = "ClusteringAlgorithm";

	Logger L = LoggerFactory.getLogger(MlplanClusterSearch.class);


	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException {
		try {
			System.out.println("ExperimentEntry: " + experimentEntry);
			final Experiment experiment = experimentEntry.getExperiment();
			System.out.println("Conduct experiment " + experiment);
			final Map<String, String> experimentDescription = experiment.getValuesOfKeyFields();
			System.out.println("Experiment Description as follows: " + experimentDescription);
			final int numCPUs = experiment.getNumCPUs();
			final ExecutorService executorService = Executors.newFixedThreadPool(numCPUs);

			try (final SQLAdapter adapter = new SQLAdapter(CONFIG.getDBHost(), CONFIG.getDBUsername(), CONFIG.getDBPassword(), CONFIG.getDBDatabaseName())) {
				final int seed = Integer.parseInt(experimentDescription.get("seed"));
				final Random rand = new Random(seed);
				//final FileReader fileReader = new FileReader(new File(CONFIG.datasetDirectory(), experimentDescription.get("dataset") + ".arff"));
				//final Instances unprocessedData = new Instances(fileReader);
				final Instances unprocessedData = OpenMLHelper.getInstancesById(Integer.parseInt(experimentDescription.get("dataset")));
				// ScikitLearnWrapper does not support nominal attributes, the NominalToNumeric Filter is used to convert the dataset
				//TODO MISSING DATA? DELETE ENTIRE ROW? -> no dataset with incomplete instances
				unprocessedData.removeIf(Instance::hasMissingValue);
				if (unprocessedData.size() == 0) {
					System.out.println("Dataset size is 0, aborting");
					return;
				}
				/* Convert nominal data to numeric */
				final Filter filter = new NominalToBinary();
				filter.setInputFormat(unprocessedData);

				filter.batchFinished();
				final Instances tmpData = Filter.useFilter(unprocessedData, filter);
				/* convert nominal class attribute to numeric */
				final Instances data;
				if (tmpData.classAttribute().isNominal()) {
					final int oldClassIndex = tmpData.classIndex();
					final Add add = new Add();
					add.setAttributeName("label");
					add.setAttributeType(new SelectedTag("NUM", Add.TAGS_TYPE));
					add.setAttributeIndex("last");
					try {
						add.setInputFormat(tmpData);
					} catch (final Exception e) {
						e.printStackTrace();
					}

					final Instances dataWithOldClass;
					try {
						dataWithOldClass = Filter.useFilter(tmpData, add);
					} catch (final Exception e) {
						e.printStackTrace();
						return;
					}
					dataWithOldClass.setClassIndex(dataWithOldClass.numAttributes() - 1);

					for (int i = 0; i < dataWithOldClass.size(); i++) {
						final String originalClassValue = tmpData.instance(i).stringValue(tmpData.classIndex());
						final double indexOfClassValue = tmpData.classAttribute().indexOfValue(originalClassValue);
						dataWithOldClass.instance(i).setValue(dataWithOldClass.classIndex(), indexOfClassValue);
					}
					final Remove removeFilter = new Remove();
					removeFilter.setAttributeIndicesArray(new int[]{oldClassIndex});
					removeFilter.setInputFormat(dataWithOldClass);
					data = Filter.useFilter(dataWithOldClass, removeFilter);
				} else {
					data = tmpData;
				}

				final Collection<Component> allComponents = new ComponentLoader(new File(FS_SEARCH_SPACE_CONFIG)).getComponents();
				final Collection<Component> componentsProvidingInterface = ComponentUtil.getComponentsProvidingInterface(allComponents, DEF_REQUESTED_HASCO_INTERFACE);
				//ADAPT HYPERPARAMETERS TO DATASET PROPERTIES
				for (final Component component : componentsProvidingInterface
				) {
					String paramName = "n_clusters";
					switch (component.getName()) {
						case "sklearn.cluster.DBSCAN":
							paramName = "min_samples";
						case "sklearn.cluster.AgglomerativeClustering":
						case "sklearn.cluster.KMeans":
							final PartialOrderedSet<Parameter> parameters = component.getParameters();
							final Parameter nClusters = component.getParameterWithName(paramName);
							parameters.remove(nClusters);
							final NumericParameterDomain domain = (NumericParameterDomain) nClusters.getDefaultDomain();
							final NumericParameterDomain newDomain = new NumericParameterDomain(domain.isInteger(), domain.getMin(), Math.min(domain.getMax(), data.size()));
							final Parameter newNCluster = new Parameter(nClusters.getName(), newDomain, nClusters.getDefaultValue());
							parameters.add(newNCluster);
					}
				}
				final List<ComponentInstance> possibleAlgorithmSelections =
					componentsProvidingInterface.stream().map(x -> new ComponentInstance(x, new HashMap<>(), new HashMap<>())).collect(Collectors.toList());

				System.out.println("Number of possible algorithm selections: " + possibleAlgorithmSelections.size());
				final SKLearnClassifierFactory factory = new SKLearnClusterClassifierFactory();

				// set timeout timer
				final Timer timer = new Timer();
				timer.schedule(new TimerTask() {
					@Override
					public void run() {
						System.out.println("Cancel mlplan search");
						status.decrementAndGet();
						executorService.shutdownNow();
						final Map<String, Object> expResult = new HashMap<>();
						expResult.put("done", 1);

						processor.processResults(expResult);
					}
				}, new TimeOut(Integer.parseInt(experimentDescription.get("timeout")), TimeUnit.SECONDS).milliseconds());
				System.out.println("Schedule jobs for evaluations.");

				final long startTimestamp = System.currentTimeMillis();

				try {
					while (status.get() == 1) {
						if (Thread.interrupted()) {
							return;
						}
						final AbstractMLPlanBuilder builder = AbstractMLPlanBuilder.forSKLearnCluster();
						builder.withTimeOut(new TimeOut(Long.parseLong(experimentDescription.get("timeout")), TimeUnit.SECONDS));
						builder.withNodeEvaluationTimeOut(new TimeOut(900, TimeUnit.SECONDS));
						builder.withCandidateEvaluationTimeOut(new TimeOut(Long.parseLong(experimentDescription.get("candTimeout")), TimeUnit.SECONDS));
						builder.withNumCpus(experimentEntry.getExperiment().getNumCPUs());
						// use internal cluster validation measure defined in experiment description
						final IInternalClusteringValidationMeasure internalMeasure;
						switch (experimentDescription.get("internalClusterValidationMeasure")) {
							case "DaviesBouldin":
								internalMeasure = new DaviesBouldinMeasure();
								break;
							case "DensityBased":
								internalMeasure = new DensityBasedMeasure();
								break;
							case "Dunn":
								internalMeasure = new DunnMeasure();
								break;
							case "Silhouette":
								internalMeasure = new SilhouetteMeasure();
								break;
							case "SquaredError":
								internalMeasure = new SquaredErrorMeasure();
								break;
							default:
								internalMeasure = new SquaredErrorMeasure();
						}

						((MLPlanSKLearnClusterBuilder) builder).withValidationMeasure(internalMeasure);
						//TODO adapt hp ranges!!!c
						final SKLearnClusterMLPlanWekaClassifier mlplan = new SKLearnClusterMLPlanWekaClassifier(builder);
						mlplan.setLoggerName("sklmlplanc");
						mlplan.setVisualizationEnabled(false);//TODO visualization
						mlplan.getMLPlanConfig().setProperty(MLPlanClassifierConfig.SELECTION_PORTION, "0");//set selection split to 0
						class ClassifierFoundListener {

							@Subscribe
							public void receiveSolutionEvent(final ClassifierFoundEvent event) {
								final Map<String, Object> results = new HashMap<>();
								results.put("experiment_id", experimentEntry.getId());
								results.put("mainClassifier", event.getSolutionCandidate().toString());
								results.put("componentInstance", event.getComponentDescription().toString());
								final double inSampleError = event.getInSampleError();
								if (new Double(inSampleError).isInfinite()) {
									if (new Double(inSampleError) > 0) {
										results.put("internalMeasureResult", Double.MAX_VALUE);
									} else {
										results.put("internalMeasureResult", Double.MIN_VALUE);
									}
								} else {
									results.put("internalMeasureResult", event.getInSampleError());
								}
								try {
									adapter.insert("mlplansearch_eval", results);
								} catch (final SQLException e) {
									e.printStackTrace();
								}
								L.info("ClassifierFoundEvent. mainClassifier: " + event.getSolutionCandidate());
							}
						}
						final ClassifierFoundListener listener = new ClassifierFoundListener();
						mlplan.registerListener(listener);

						mlplan.buildClassifier(data);

						System.out.println("Evaluate " + mlplan.getSelectedClassifier() + " " + experimentDescription.get("dataset"));

						// store experiment information
						final Map<String, Object> results = new HashMap<>();
						results.put("experiment_id", experimentEntry.getId());
						results.put("mainClassifier", mlplan.getSelectedClassifier());
						results.put("componentInstance", mlplan.getSelectedClassifier());
						System.out.println(mlplan.getSelectedClassifier());
						try {
							final long evalStart;
							{
								final Classifier c = mlplan;

								/* Validate the classifier */
								evalStart = System.currentTimeMillis();
								final ClusteringResult valRes = ClusteringEvaluation.evaluateModel(c, data, internalMeasure);

								results.put("valTrainTime", (double) (System.currentTimeMillis() - evalStart));
								if (valRes.isResultValid()) {
									results.put("internalMeasureResult", valRes.getInternalEvaluationResult());
								} else {
									results.put("internalMeasureResult", Double.MAX_VALUE); //maximal loss if invalid
								}
							}

							results.put("secondsUntilFound", (int) ((double) (System.currentTimeMillis() - startTimestamp) / 1000));
							results.put("exception", "");


						} catch (final Exception e) {
							e.printStackTrace();
							final StringBuilder stackTraceBuilder = new StringBuilder();
							for (final Throwable ex : ExceptionUtils.getThrowables(e)) {
								stackTraceBuilder.append(ExceptionUtils.getStackTrace(ex) + "\n");
							}
							results.put("exception", stackTraceBuilder.toString());
						} finally {
							final Map<String, Object> map = new HashMap<>();
							map.putAll(results);
							map.remove("experiment_id");
							processor.processResults(map);
						}

					}
				} catch (final Throwable e) {
					e.printStackTrace();
				}


			}
		} catch (final Exception e) {
			e.printStackTrace();
			throw new ExperimentEvaluationFailedException(e);
		}

	}


	public static void main(final String[] args) throws ExperimentDBInteractionFailedException, IllegalExperimentSetupException {
		//CONFIG.setProperty("db.table", "experiment002");
		System.out.println(CONFIG);
		final ExperimentRunner runner = new ExperimentRunner(CONFIG, new MlplanClusterSearch(), new ExperimenterSQLHandle(CONFIG));//, 1);
		runner.randomlyConductExperiments(1, false);
	}

}