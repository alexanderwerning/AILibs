package ai.libs.mlplan.unlabeled;

import ai.libs.hasco.model.BooleanParameterDomain;
import ai.libs.hasco.model.CategoricalParameterDomain;
import ai.libs.hasco.model.Component;
import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.model.ComponentUtil;
import ai.libs.hasco.model.NumericParameterDomain;
import ai.libs.hasco.model.Parameter;
import ai.libs.hasco.model.ParameterRefinementConfiguration;
import ai.libs.hasco.serialization.ComponentLoader;
import ai.libs.jaicore.basic.FileUtil;
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
import ai.libs.jaicore.logging.ToJSONStringUtil;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.DaviesBouldinMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.DensityBasedMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.DunnMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.IInternalClusteringValidationMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.SilhouetteMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.SquaredErrorMeasure;
import ai.libs.mlplan.core.AbstractMLPlanBuilder;
import ai.libs.mlplan.core.MLPlanSKLearnClusterBuilder;
import ai.libs.mlplan.core.events.ClassifierFoundEvent;
import ai.libs.mlplan.multiclass.MLPlanClassifierConfig;
import ai.libs.mlplan.multiclass.wekamlplan.sklearn.SKLearnClassifierFactory;
import ai.libs.mlplan.multiclass.wekamlplan.sklearn.SKLearnClusterClassifierFactory;
import ai.libs.mlplan.multiclass.wekamlplan.sklearn.SKLearnClusterMLPlanWekaClassifier;
import ai.libs.mlplan.unlabeled.ClusteringEvaluation.ClusteringResult;
import com.google.common.eventbus.Subscribe;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ConcurrentHashMap;
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
	private static final String FS_SEARCH_SPACE_CONFIG = "conf/sklearn-cluster-mlplan.json";
	private static final String RES_SKLEARN_SEARCHSPACE_CONFIG = "automl/searchmodels/sklearn/sklearn-cluster-mlplan.json";
	private static final String DEF_REQUESTED_HASCO_INTERFACE = "ClusteringAlgorithm";

	private final Logger L = LoggerFactory.getLogger(MlplanClusterSearch.class);

	private static final File TMP_FOLDER = new File("tmp"); // Folder to put the json files

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException {
		try {
			this.L.info("ExperimentEntry: " + experimentEntry);
			final Experiment experiment = experimentEntry.getExperiment();
			this.L.info("Conduct experiment " + experiment);
			final Map<String, String> experimentDescription = experiment.getValuesOfKeyFields();
			this.L.info("Experiment Description as follows: " + experimentDescription);
			final int numCPUs = experiment.getNumCPUs();

			try (final SQLAdapter adapter = new SQLAdapter(CONFIG.getDBHost(), CONFIG.getDBUsername(), CONFIG.getDBPassword(), CONFIG.getDBDatabaseName())) {
				final int seed = Integer.parseInt(experimentDescription.get("seed"));
				final Random rand = new Random(seed);
				final FileReader fileReader = new FileReader(new File(CONFIG.datasetDirectory(), experimentDescription.get("dataset")));
				final Instances unprocessedData = new Instances(fileReader);
				//final Instances unprocessedData = OpenMLHelper.getInstancesById(Integer.parseInt(experimentDescription.get("dataset")));
				/* ScikitLearnWrapper does not support nominal attributes, the NominalToNumeric Filter is used to convert the dataset*/
				unprocessedData.removeIf(Instance::hasMissingValue);
				if (unprocessedData.size() == 0) {
					this.L.error("Dataset size is 0, aborting");
					return;
				}
				/* Convert nominal data to numeric */
				final Filter filter = new NominalToBinary();
				filter.setInputFormat(unprocessedData);

				filter.batchFinished();
				final Instances tmpData = Filter.useFilter(unprocessedData, filter);
				/* convert nominal class attribute to numeric */
				final Instances data;
				if (tmpData.classIndex() != -1 && tmpData.classAttribute().isNominal()) {
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

				final ComponentLoader loader = new ComponentLoader(FileUtil.getExistingFileWithHighestPriority(RES_SKLEARN_SEARCHSPACE_CONFIG, FS_SEARCH_SPACE_CONFIG));
				final Collection<Component> allComponents = loader.getComponents();
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

				this.L.info("Number of possible algorithm selections: " + possibleAlgorithmSelections.size());
				final SKLearnClassifierFactory factory = new SKLearnClusterClassifierFactory();

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
				final Timer timer = new Timer();
				final long startTimestamp;

				switch (experimentDescription.get("strategy")) {
					case "mlplan":

						startTimestamp = System.currentTimeMillis();

						try {

							final AbstractMLPlanBuilder builder = AbstractMLPlanBuilder.forSKLearnCluster();
							builder.withTimeOut(new TimeOut(Long.parseLong(experimentDescription.get("timeout")), TimeUnit.SECONDS));
							builder.withNodeEvaluationTimeOut(new TimeOut(900, TimeUnit.SECONDS));
							builder.withCandidateEvaluationTimeOut(new TimeOut(Long.parseLong(experimentDescription.get("candTimeout")), TimeUnit.SECONDS));
							builder.withNumCpus(experimentEntry.getExperiment().getNumCPUs());

							// TODO reading the files creates error on pc2, but it would not change any values for given datasets anyway
							//final File createdFile = getSearchSpaceConfigFile(componentsProvidingInterface, loader.getParamConfigs());
							//builder.withSearchSpaceConfigFile(createdFile);

							((MLPlanSKLearnClusterBuilder) builder).withValidationMeasure(internalMeasure);
							final SKLearnClusterMLPlanWekaClassifier mlplan = new SKLearnClusterMLPlanWekaClassifier(builder);
							mlplan.setLoggerName("AutomatedClusteringLogger");
							mlplan.setVisualizationEnabled(false);
							mlplan.getMLPlanConfig().setProperty(MLPlanClassifierConfig.SELECTION_PORTION, "0");//set selection split to 0
							class ClassifierFoundListener {

								@Subscribe
								public void receiveSolutionEvent(final ClassifierFoundEvent event) {
									final Map<String, Object> results = new HashMap<>();
									results.put("experiment_id", experimentEntry.getId());
									results.put("mainClassifier", event.getSolutionCandidate().toString());
									results.put("componentInstance", event.getComponentDescription().toString());
									results.put("secondsUntilFound", (int) ((double) (System.currentTimeMillis() - startTimestamp) / 1000));
									final double inSampleError = event.getInSampleError();
									final long evalStart = System.currentTimeMillis();
									final ClusteringResult valRes = ClusteringEvaluation.evaluateModel(event.getSolutionCandidate(), data, internalMeasure);

									results.put("valTrainTime", (double) (System.currentTimeMillis() - evalStart) / 1000);

									if (new Double(inSampleError).isInfinite()) {
										if (new Double(inSampleError) > 0) {
											results.put("internalMeasureResult", Double.MAX_VALUE);
										} else {
											results.put("internalMeasureResult", Double.MIN_VALUE);
										}
									} else {
										results.put("internalMeasureResult", event.getInSampleError());
									}
									results.put("externalMeasureResult", valRes.getExternalEvaluationResult());
									results.put("n_clusters", valRes.getN_clusters());
									results.put("exception", "");
									results.put("done", 1);
									try {
										adapter.insert(CONFIG.resultsTable(), results);
									} catch (final SQLException e) {
										e.printStackTrace();
									}
								}
							}
							final ClassifierFoundListener listener = new ClassifierFoundListener();
							mlplan.registerListener(listener);

							mlplan.buildClassifier(data);

							this.L.info("Evaluate " + mlplan.getSelectedClassifier() + " " + experimentDescription.get("dataset"));

							// store experiment information
							final Map<String, Object> results = new HashMap<>();
							results.put("mainClassifier", mlplan.getSelectedClassifier());
							results.put("componentInstance", mlplan.getSelectedClassifier());

							try {
								final long evalStart;
								{
									results.put("secondsUntilFound", (int) ((double) (System.currentTimeMillis() - startTimestamp) / 1000));
									final Classifier c = mlplan;

									/* Validate the classifier */
									evalStart = System.currentTimeMillis();
									final ClusteringResult valRes = ClusteringEvaluation.evaluateModel(c, data, internalMeasure);

									results.put("valTrainTime", (double) (System.currentTimeMillis() - evalStart) / 1000);
									if (!((Double) valRes.getInternalEvaluationResult()).isInfinite() && !((Double) valRes.getInternalEvaluationResult()).isNaN()) {
										results.put("internalMeasureResult", valRes.getInternalEvaluationResult());

									} else {
										results.put("internalMeasureResult", Double.MAX_VALUE); //maximal loss if invalid

									}
									results.put("externalMeasureResult", valRes.getExternalEvaluationResult());
									results.put("n_clusters", valRes.getN_clusters());
								}

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
								map.put("done", 1);
								processor.processResults(map);
							}


						} catch (final Throwable e) {
							e.printStackTrace();
						}

						break;
					case "random":
						final TimeOut candidateTimeout = new TimeOut(Integer.parseInt(experimentDescription.get("candTimeout")), TimeUnit.SECONDS);

						final ConcurrentHashMap<String, Object> bestResult = new ConcurrentHashMap<>();

						// set timeout timer

						timer.schedule(new TimerTask() {
							@Override
							public void run() {
								System.out.println("Cancel random search");
								status.decrementAndGet();
								final Map<String, Object> expResult = new HashMap<>();
								expResult.put("done", 1);
								expResult.putAll(bestResult);
								expResult.remove("experiment_id");
								processor.processResults(expResult);
							}
						}, new TimeOut(Integer.parseInt(experimentDescription.get("timeout")), TimeUnit.SECONDS).milliseconds());
						this.L.info("Schedule jobs for evaluations.");
						startTimestamp = System.currentTimeMillis();
						try {
							while (status.get() == 1) {
								if (Thread.interrupted()) {
									return;
								}
								// select random algorithm
								final ComponentInstance ci = possibleAlgorithmSelections.get(rand.nextInt(possibleAlgorithmSelections.size()));
								final LinkedList<ComponentInstance> cisForRandomParams = new LinkedList<>();
								cisForRandomParams.add(ci);
								// parametrize algorithm
								while (!cisForRandomParams.isEmpty()) {
									final ComponentInstance ciHyp = cisForRandomParams.poll();
									cisForRandomParams.addAll(ciHyp.getSatisfactionOfRequiredInterfaces().values());
									ciHyp.getParameterValues().clear();
									ciHyp.getParameterValues().putAll(ComponentUtil.randomParameterizationOfComponent(ciHyp.getComponent(), rand).getParameterValues());
								}

								// set candidate timeout
								final TimerTask candidateTimeoutTask = new TimerTask() {
									private final Thread currentThread = Thread.currentThread();

									@Override
									public void run() {
										MlplanClusterSearch.this.L.info("Candidate Timeout: Interrupt current thread evaluating " + ci.getNestedComponentDescription());
										this.currentThread.interrupt();
									}
								};
								timer.schedule(candidateTimeoutTask, candidateTimeout.milliseconds());

								MlplanClusterSearch.this.L.info("Evaluate " + ci.getNestedComponentDescription() + " " + experimentDescription.get("dataset"));

								// store experiment information
								final Map<String, Object> results = new HashMap<>();
								results.put("experiment_id", experimentEntry.getId());
								results.put("mainClassifier", ci.getNestedComponentDescription());
								results.put("componentInstance", ci.toString());
								try {
									final long evalStart;
									{
										results.put("secondsUntilFound", (int) ((double) (System.currentTimeMillis() - startTimestamp) / 1000));
										final Classifier c = factory.getComponentInstantiation(ci);

										/* Validate the classifier */
										evalStart = System.currentTimeMillis();
										final ClusteringResult valRes = ClusteringEvaluation.evaluateModel(c, data, internalMeasure);

										results.put("valTrainTime", (double) (System.currentTimeMillis() - evalStart) / 1000);
										if (!((Double) valRes.getInternalEvaluationResult()).isInfinite() && !((Double) valRes.getInternalEvaluationResult()).isNaN()) {
											results.put("internalMeasureResult", valRes.getInternalEvaluationResult());

										} else {
											results.put("internalMeasureResult", Double.MAX_VALUE); //maximal loss if invalid

										}
										results.put("externalMeasureResult", valRes.getExternalEvaluationResult());
										results.put("n_clusters", valRes.getN_clusters());
									}

									results.put("exception", "");
								} catch (final Exception e) {
									e.printStackTrace();
									final StringBuilder stackTraceBuilder = new StringBuilder();
									for (final Throwable ex : ExceptionUtils.getThrowables(e)) {
										stackTraceBuilder.append(ExceptionUtils.getStackTrace(ex) + "\n");
									}
									results.put("exception", stackTraceBuilder.toString());
								} finally {
									candidateTimeoutTask.cancel();
								}
								try {
									results.put("done", 1);
									adapter.insert(CONFIG.resultsTable(), results);
									if (bestResult.size() == 0 || (double) results.get("internalMeasureResult") < (double) bestResult.get("internalMeasureResult")) {
										bestResult.clear();
										bestResult.putAll(results);
									}
									results.remove("componentInstance");
									MlplanClusterSearch.this.L.info("Results collected " + results);
								} catch (final SQLException e) {
									e.printStackTrace();
								}

							}
						} catch (final Throwable e) {
							e.printStackTrace();
						}

						break;
				}
			}
		} catch (final Exception e) {
			e.printStackTrace();
			throw new ExperimentEvaluationFailedException(e);
		}

	}

	private File getSearchSpaceConfigFile(final Collection<Component> components, final Map<Component, Map<Parameter, ParameterRefinementConfiguration>> paramConfigs) {
		final String filename = "searchSpace" + components.hashCode() + ".json";

		final File outputFile = new File(TMP_FOLDER, filename);
		if (outputFile.exists()) {
			this.L.debug("Reusing {}", outputFile);
			return outputFile;
		}
		try (final BufferedWriter bw = new BufferedWriter(new FileWriter(outputFile))) {
			bw.write(serializeComponents(components, paramConfigs));
		} catch (final IOException e) {
			this.L.error("could not write temporary search space config");
		}
		return outputFile;
	}

	private String serializeComponents(final Collection<Component> components, final Map<Component, Map<Parameter, ParameterRefinementConfiguration>> paramConfigs) {
		final StringBuilder sb = new StringBuilder();
		sb.append("{\n"
					  + "  \"repository\" : \"ClusterProblem\",\n"
					  + "  \"components\" : [ ");
		final Iterator<Component> it = components.iterator();

		while (it.hasNext()) {
			final Component c = it.next();
			sb.append(serializeComponent(c, paramConfigs.get(c)));
			if (it.hasNext()) {
				sb.append(",");
			}
		}
		sb.append("]}");
		return sb.toString();
	}

	private String serializeComponent(final Component component, final Map<Parameter, ParameterRefinementConfiguration> paramConfig) {

		final Map<String, Object> fields = new HashMap<>();
		fields.put("name", component.getName());
		fields.put("providedInterface", component.getProvidedInterfaces());
		fields.put("requiredInterface", component.getRequiredInterfaces());
		final ArrayList<Map<String, Object>> params = new ArrayList<>();
		for (final Parameter p : component.getParameters()) {
			final HashMap<String, Object> param = new HashMap<>();
			param.put("name", p.getName());
			if (p.getDefaultDomain() instanceof NumericParameterDomain) {
				final NumericParameterDomain domain = (NumericParameterDomain) p.getDefaultDomain();
				if (domain.isInteger()) {
					param.put("type", "int");
					param.put("min", ((Double) domain.getMin()).intValue());
					param.put("max", ((Double) domain.getMax()).intValue());
					param.put("default", ((Double) p.getDefaultValue()).intValue());
				} else {
					param.put("type", "double");
					param.put("min", domain.getMin());
					param.put("max", domain.getMax());
					param.put("default", p.getDefaultValue());
				}
			} else if (p.getDefaultDomain() instanceof CategoricalParameterDomain) {
				final CategoricalParameterDomain domain = (CategoricalParameterDomain) p.getDefaultDomain();
				param.put("type", "cat");
				param.put("values", domain.getValues());
				param.put("default", p.getDefaultValue());
			} else if (p.getDefaultDomain() instanceof BooleanParameterDomain) {
				final BooleanParameterDomain domain = (BooleanParameterDomain) p.getDefaultDomain();
				param.put("type", "boolean");
				param.put("default", p.getDefaultValue());
			}
			if (paramConfig.containsKey(p)) {
				final ParameterRefinementConfiguration config = paramConfig.get(p);
				param.put("refineSplits", config.getRefinementsPerStep());
				if (p.getDefaultDomain() instanceof NumericParameterDomain && ((NumericParameterDomain) p.getDefaultDomain()).isInteger()) {
					param.put("minInterval", ((Double) config.getIntervalLength()).intValue());
				} else {
					param.put("minInterval", config.getIntervalLength());
				}
				param.put("includeExtremals", config.isInitWithExtremalPoints());
			}
			params.add(param);
		}

		fields.put("parameter", params);
		return ToJSONStringUtil.toJSONString(fields);
	}

	public static void main(final String[] args) throws ExperimentDBInteractionFailedException, IllegalExperimentSetupException {

		CONFIG.setProperty("dataset", Arrays.stream(CONFIG.datasetDirectory().list()).filter(x -> x.endsWith(".arff")).collect(Collectors.joining(",")));
		final ExperimentRunner runner = new ExperimentRunner(CONFIG, new MlplanClusterSearch(), new ExperimenterSQLHandle(CONFIG));
		runner.randomlyConductExperiments(1, false);

	}

}