package wekaImages.dataMining;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import javax.swing.plaf.synth.SynthSeparatorUI;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.clusterers.Clusterer;
import weka.clusterers.EM;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.imagefilter.AbstractImageFilter;
import weka.filters.unsupervised.instance.imagefilter.AutoColorCorrelogramFilter;
import weka.filters.unsupervised.instance.imagefilter.BinaryPatternsPyramidFilter;
import weka.filters.unsupervised.instance.imagefilter.ColorLayoutFilter;
import weka.filters.unsupervised.instance.imagefilter.EdgeHistogramFilter;
import weka.filters.unsupervised.instance.imagefilter.FCTHFilter;
import weka.filters.unsupervised.instance.imagefilter.FuzzyOpponentHistogramFilter;
import weka.filters.unsupervised.instance.imagefilter.GaborFilter;
import weka.filters.unsupervised.instance.imagefilter.JpegCoefficientFilter;
import weka.filters.unsupervised.instance.imagefilter.PHOGFilter;
import weka.filters.unsupervised.instance.imagefilter.SimpleColorHistogramFilter;
import wekaImages.control.IncompatibleAttributeException;
import wekaImages.control.MissingModelDataException;

/**
 * Class that implements the facade pattern
 * 
 * Its methods provide the main functionality of the program using the classes
 * that are necessary
 * 
 * @author JosÃ© Francisco DÃ­ez
 * 
 */
public class DataMiningFacade {

	private static DataMiningFacade facade;

	private Instances dataset;
	private Classifier classifier;
	private Clusterer clusterer;

	/**
	 * Obtain the static instance of DataMiningFacade PD Singleton
	 * 
	 * @return instance
	 */
	public static DataMiningFacade getFacade() {

		if (facade == null) {

			facade = new DataMiningFacade();
		}
		return facade;
	}

	/**
	 * Private constructor PD Singleton
	 */
	private DataMiningFacade() {

	}

	// ////////////
	/**
	 * Load a dataset stored in the file system using the absolute path
	 * 
	 * @param path
	 *            arff absolute path
	 */
	public void loadDataset(String path) {
		DataSource source;
		try {
			// Cargamos el arff en source
			source = new DataSource(path);

			// Creamos una instancia en la que guardamos el dataSet
			dataset = source.getDataSet();

			// si no usamos XRFF es necesario especificar la clase
			if (dataset.classIndex() == -1)
				dataset.setClassIndex(dataset.numAttributes() - 1);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Load a classifier stored in the file system using the absolute path
	 * 
	 * @param path
	 *            arff absolute path
	 */
	public void loadClassifier(String path) {

		try {
			// Deserializamos el clasificador
			classifier = (Classifier) weka.core.SerializationHelper.read(path);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Load a clusterer stored in the file system using the absolute path
	 * 
	 * @param path
	 *            arff absolute path
	 */
	public void loadClusterer(String path) {
		try {
			// Deserializamos el cluster
			clusterer = (Clusterer) weka.core.SerializationHelper.read(path);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Saves the classifier in a fixed location
	 */
	public void saveClassifier() {
		try {
			// Serializamos el cluster
			weka.core.SerializationHelper.write("clasificador.model", classifier);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Saves the clusteter in a fixed location
	 */
	public void saveClusterer() {
		try {
			// Serializamos el cluster
			weka.core.SerializationHelper.write("clusterer.model", clusterer);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Saves the dataset in a fixed location
	 */
	public void saveDataset() {
		try {

			// Creamos el fichero
			File datasetFile = new File(".", "dataset.arff");

			// Creamos un objeto que escribe en el fichero
			BufferedWriter writer = new BufferedWriter(new FileWriter(datasetFile));

			// Escribimos en el fichero
			writer.write(dataset.toString());
			writer.flush();
			writer.close();

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * This methods trains a new classifier
	 * 
	 * @param imagePathsMap
	 *            a hash table that has as many keys as classes and each key is
	 *            associated with an arraylist with the absolute paths of all
	 *            images associated with that class.
	 * @param options
	 *            boolean array with the image filters to apply
	 */
	public void trainClassiffier(HashMap<String, ArrayList<String>> imagePathsMap, boolean[] options) {

		try {

			// Creamos el dataset correspondientes a los valores del mapa y de
			// las opciones
			buildDataset(imagePathsMap, options);

			Remove remove = new Remove();

			remove.setAttributeIndices("1");

			remove.setInputFormat(dataset);

			Instances datasetCopia = Filter.useFilter(dataset, remove);

			// Los clasificadores son objetos java que heredan de
			// AbstractClassifier
			classifier = new J48();

			// Para entrenar el clasificador usamos buildClassifier
			classifier.buildClassifier(datasetCopia);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * This methods trains a new clusterer
	 * 
	 * @param imagePathsMap
	 *            a hash table that has as many keys as classes and each key is
	 *            associated with an arraylist with the absolute paths of all
	 *            images associated with that class.
	 * @param options
	 *            boolean array with the image filters to apply
	 */
	public void trainClusterer(HashMap<String, ArrayList<String>> imagePathsMap, boolean[] options) {
		try {
			// Creamos el dataset correspondientes a los valores del mapa y de
			// las opciones
			buildDataset(imagePathsMap, options);

			Remove remove = new Remove();

			remove.setAttributeIndices("1,last");

			remove.setInputFormat(dataset);

			Instances datasetCopia = Filter.useFilter(dataset, remove);

			clusterer = new EM(); // new instance of clusterer

			// Para entrenar el clasificador usamos buildClassifier
			clusterer.buildClusterer(datasetCopia);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Función que devuleve un array con los filtros a aplicar
	 * 
	 * @param op
	 *            opciones indicando que filtros se aplican.
	 * @return array de filtros que contiene todos los filtros a aplicar
	 */
	public Filter[] añadeFiltros(boolean[] op) {
		int cont = 0;
		int tam = 0;

		// Calculamos el tamaño de arrayFiltro a partir del numero de trues
		for (boolean b : op) {
			if (b)
				tam++;
		}

		AbstractImageFilter[] arrayFiltros = new AbstractImageFilter[tam];

		for (int i = 0; i < op.length; i++) {
			if (op[i]) {
				switch (i) {
				case 0:
					arrayFiltros[cont] = new AutoColorCorrelogramFilter();

					break;
				case 1:
					arrayFiltros[cont] = new BinaryPatternsPyramidFilter();
					break;
				case 2:
					arrayFiltros[cont] = new ColorLayoutFilter();
					break;
				case 3:
					arrayFiltros[cont] = new EdgeHistogramFilter();
					break;
				case 4:
					arrayFiltros[cont] = new FCTHFilter();
					break;
				case 5:
					arrayFiltros[cont] = new FuzzyOpponentHistogramFilter();
					break;
				case 6:
					arrayFiltros[cont] = new GaborFilter();
					// arrayFiltros[cont].setImageDirectory("");

					break;
				case 7:
					arrayFiltros[cont] = new JpegCoefficientFilter();
					break;
				case 8:
					arrayFiltros[cont] = new PHOGFilter();
					break;
				case 9:
					arrayFiltros[cont] = new SimpleColorHistogramFilter();
					break;
				}

				cont++;
			}
		}
		return arrayFiltros;
	}

	/**
	 * This methods creates a new dataset
	 * 
	 * @param imagePathsMap
	 *            a hash table that has as many keys as classes and each key is
	 *            associated with an arraylist with the absolute paths of all
	 *            images associated with that class.
	 * @param options
	 *            boolean array with the image filters to apply
	 */
	public void buildDataset(HashMap<String, ArrayList<String>> imagePathsMap, boolean[] options) {

		try {
			String path = "C:\\Users\\usuario\\Desktop\\Ingenieria Informatica\\Zuarto\\Segundo Semestre\\Minería de Datos\\Practica\\WekaProgramming";

			// Creamos el multifilter
			MultiFilter multifilter = new MultiFilter();

			// Añadimos todos los filtros al multifilter
			Filter[] filtros = añadeFiltros(options);
			multifilter.setFilters(filtros);

			// Creamos el fichero
			File datasetFile = new File(path, "prueba.arff");

			// Creamos un objeto que escribe en el fichero
			BufferedWriter writer = new BufferedWriter(new FileWriter(datasetFile));

			String imgDir = System.getProperty("last.dir", System.getProperty("user.dir"));

			// Escribimos la primera línea
			writer.write("@relation dataset\n");

			// Escribimos el filename
			writer.write("@attribute filename string\n");

			// Escribimos las clases
			writer.write(
					"@attribute class {"
							+ imagePathsMap.keySet().toString()
									.substring(1, imagePathsMap.keySet().toString().length() - 1).replace(", ", ",")
							+ "}\n");
			writer.write("@data");

			// Escribimos el resto de los datos
			for (String key : imagePathsMap.keySet()) {
				for (String v : imagePathsMap.get(key)) {
					String ruta = new File(imgDir).toURI().relativize(new File(v).toURI()).getPath();
					writer.write("\n" + ruta + ", " + key);
				}
			}

			// Cerramos el fichero
			writer.flush();
			writer.close();

			// Escribimos en la variable dataset
			DataSource source = new DataSource(datasetFile.getCanonicalPath());

			// Extraemos los datos temporales
			Instances tmpData = source.getDataSet();

			// Se le informa al filtro de cual es el dataset, DESPUES de
			// establecer las opciones
			multifilter.setInputFormat(tmpData);

			// Creamos el dataset definitivo una vez que le hemos pasado todos
			// los filtros
			dataset = MultiFilter.useFilter(tmpData, multifilter);

			// Le decimos cual es su clase
			dataset.setClassIndex(dataset.numAttributes() - 1);

		} catch (IOException e) {
			e.printStackTrace();

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * This method predicts the class of an image, using the features indicated
	 * in options and the classifier already built
	 * 
	 * @param imagePath
	 *            of the image file
	 * @param options
	 *            of the image filters
	 * @return the class name
	 * @throws MissingModelDataException
	 *             the classifier is not built
	 * @throws IncompatibleAttributeException
	 *             the options are not compatible with the already built
	 *             classifier
	 */
	public String predictImage(String imagePath, boolean[] options)
			throws MissingModelDataException, IncompatibleAttributeException {

		String retorno = null;

		try {
			String path = "C:\\Users\\usuario\\Desktop\\Ingenieria Informatica\\Zuarto\\Segundo Semestre\\Minería de Datos\\Practica\\WekaProgramming";

			// Creamos el multifilter
			MultiFilter multifilter = new MultiFilter();

			// Añadimos todos los filtros al multifilter
			Filter[] filtros = añadeFiltros(options);
			multifilter.setFilters(filtros);

			// Creamos el fichero
			File datasetFile = new File(path, "prueba.arff");

			// Creamos un objeto que escribe en el fichero
			BufferedWriter writer = new BufferedWriter(new FileWriter(datasetFile));

			String imgDir = System.getProperty("last.dir", System.getProperty("user.dir"));

			// Escribimos la primera línea
			writer.write("@relation dataset\n");

			// Escribimos el filename
			writer.write("@attribute filename string\n");

			writer.write("@data\n");

			// Calculamos la ruta
			String ruta = new File(imgDir).toURI().relativize(new File(imagePath).toURI()).getPath();

			// Escribimos el resto de los datos
			writer.write(ruta);

			// Cerramos el fichero
			writer.flush();
			writer.close();

			// Escribimos en la variable dataset
			DataSource source = new DataSource(datasetFile.getCanonicalPath());

			// Extraemos los datos temporales
			Instances tmpData = source.getDataSet();

			// Se le informa al filtro de cual es el dataset, DESPUES de
			// establecer las opciones
			multifilter.setInputFormat(tmpData);

			// Creamos el dataset definitivo una vez que le hemos pasado todos
			// los filtros
			Instances datasetCopia = Filter.useFilter(tmpData, multifilter);

			// Le decimos cual es su clase
			datasetCopia.setClassIndex(datasetCopia.numAttributes() - 1);

			// Creamos la instancia con la que entrenar el clasificador
			Instance ins = datasetCopia.get(0);

			// Obtenemos el indice de la clase a la que corresponde la instancia
			double indice = classifier.classifyInstance(ins);

			// Devolvemos el nombre de la clase a la que pertenece
			retorno = dataset.classAttribute().value((int) indice);

		} catch (IOException e) {
			e.printStackTrace();

		} catch (Exception e) {
			e.printStackTrace();
		}
		return retorno;
	}

	/**
	 * This method find the most similar images using the features indicated in
	 * options and the dataset already built
	 * 
	 * @param imagePath
	 *            of the image file
	 * @param options
	 *            of the image filters
	 * @return A hashmap with one key and the list of image paths associated to
	 *         this key
	 * @throws MissingModelDataException
	 *             the classifier is not built
	 * @throws IncompatibleAttributeException
	 *             the options are not compatible with the already built
	 *             classifier
	 */
	public HashMap<String, ArrayList<String>> findSimilarities(String imagePath, boolean[] options)
			throws MissingModelDataException, IncompatibleAttributeException {

		HashMap<String, ArrayList<String>> retorno = new HashMap<String, ArrayList<String>>();
		ArrayList<String> listaVecinos = new ArrayList<String>();
		Instances data = null;

		try {
			String path = "C:\\Users\\usuario\\Desktop\\Ingenieria Informatica\\Zuarto\\Segundo Semestre\\Minería de Datos\\Practica\\WekaProgramming";

			// Creamos el multifilter
			MultiFilter multifilter = new MultiFilter();

			// Añadimos todos los filtros al multifilter
			Filter[] filtros = añadeFiltros(options);
			multifilter.setFilters(filtros);

			// Creamos el fichero
			File datasetFile = new File(path, "prueba.arff");

			// Creamos un objeto que escribe en el fichero
			BufferedWriter writer = new BufferedWriter(new FileWriter(datasetFile));

			String imgDir = System.getProperty("last.dir", System.getProperty("user.dir"));

			// Escribimos la primera línea
			writer.write("@relation dataset\n");

			// Escribimos el filename
			writer.write("@attribute filename string\n");

			writer.write("@data\n");

			// Calculamos la ruta
			String ruta = new File(imgDir).toURI().relativize(new File(imagePath).toURI()).getPath();

			// Escribimos el resto de los datos
			writer.write(ruta);

			// Cerramos el fichero
			writer.flush();
			writer.close();

			// Escribimos en la variable dataset
			DataSource source = new DataSource(datasetFile.getCanonicalPath());

			// Extraemos los datos temporales
			Instances tmpData = source.getDataSet();

			// Se le informa al filtro de cual es el dataset, DESPUES de
			// establecer las opciones
			multifilter.setInputFormat(tmpData);

			// Creamos el dataset definitivo una vez que le hemos pasado todos
			// los filtros
			data = Filter.useFilter(tmpData, multifilter);

			// Le decimos cual es su clase
			dataset.setClassIndex(dataset.numAttributes() - 1);

			// Creamos la instancia con la que entrenar el clasificador
			Instance ins = data.get(0);

			// Crea el método de búsqueda, otras opciones son BallTree,
			// CoverTree, KDTree
			NearestNeighbourSearch NNSearch = new LinearNNSearch();

			// ChebyshevDistance, EditDistance, EuclideanDistance,
			// ManhattanDistance, NormalizableDistance
			// NNSearch.setDistanceFunction(new ManhattanDistance());
			NNSearch.setInstances(dataset);

			// Obtiene los 5 vecinos más cercanos
			Instances neighbours = NNSearch.kNearestNeighbours(ins, 5);

			for (int i = 0; i < neighbours.size(); i++) {
				int index = (int) neighbours.get(i).value(0);
				String rut = neighbours.get(i).attribute(0).value(index);
				listaVecinos.add(rut);
			}

			retorno.put("", listaVecinos);

		} catch (IOException e) {
			e.printStackTrace();

		} catch (Exception e) {
			e.printStackTrace();
		}
		return retorno;

	}

	/**
	 * This method shows cluster assigments of the images contained in the
	 * dataset
	 * 
	 * @return A hashmap with one key per cluster and the list of image paths
	 *         that belongs to each cluster associated to this key
	 * @throws MissingModelDataException
	 *             the classifier is not built
	 * @throws IncompatibleAttributeException
	 *             the options are not compatible with the already built
	 *             classifier
	 */
	public HashMap<String, ArrayList<String>> viewClusters()
			throws MissingModelDataException, IncompatibleAttributeException {

		HashMap<String, ArrayList<String>> mapa = new HashMap<String, ArrayList<String>>();

		try {

			Remove remove = new Remove();

			remove.setAttributeIndices("1,last");

			remove.setInputFormat(dataset);

			Instances datasetCluster = Filter.useFilter(dataset, remove);

			int numClus = ((EM) clusterer).getNumClusters();

			System.out.println(datasetCluster);

			for (int i = 0; i < datasetCluster.size(); i++) {

				if (mapa.containsKey("cluster " + clusterer.clusterInstance(datasetCluster.get(i)))) {

					// Añadimos la imagen al clusterer correspondiente
					mapa.get("cluster " + clusterer.clusterInstance(datasetCluster.get(i)))
							.add(dataset.get(i).stringValue(0));
				} else {

					// Como no existe, creamos un ArrayList
					ArrayList<String> lista = new ArrayList<String>();

					// Le introducimos la ruta concreta de la imagen
					lista.add(dataset.get(i).stringValue(0));

					// Introducimos el clusterer al mapa junto con la lista
					// creada anteriormente
					mapa.put("cluster " + clusterer.clusterInstance(datasetCluster.get(i)), lista);
				}
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		return mapa;

	}

}