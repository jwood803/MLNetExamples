using Microsoft.ML;
using Microsoft.ML.AutoML;
using System;

namespace InferColumns
{
    class Program
    {
        private static string FILE_PATH = "./housing.csv";
        private static string LABEL_NAME = "median_house_value";

        static void Main(string[] args)
        {
            var context = new MLContext();

            // Inferring with column information
            //var columnInfo = new ColumnInformation
            //{
            //    LabelColumnName = LABEL_NAME
            //};
            //var inference = context.Auto().InferColumns(FILE_PATH, columnInfo, separatorChar: ',');

            // Inferring with label column index
            //var inference = context.Auto().InferColumns(FILE_PATH, labelColumnIndex: 8, hasHeader: true, separatorChar: ',');
            
            // Inferring with label column name
            var inference = context.Auto().InferColumns(FILE_PATH, labelColumnName: LABEL_NAME, separatorChar: ',');

            var loader = context.Data.CreateTextLoader(inference.TextLoaderOptions);

            var data = loader.Load(FILE_PATH);

            var split = context.Data.TrainTestSplit(data, testFraction: 0.2);

            var experimentSettings = new RegressionExperimentSettings
            {
                MaxExperimentTimeInSeconds = 60,
                OptimizingMetric = RegressionMetric.RSquared
            };

            var experimentResult = context.Auto()
                .CreateRegressionExperiment(experimentSettings)
                .Execute(split.TrainSet, labelColumnName: LABEL_NAME);

            var predictions = experimentResult.BestRun.Model.Transform(split.TestSet);

            var metrics = context.Regression.Evaluate(predictions, LABEL_NAME);

            Console.WriteLine($"R^2: {metrics.RSquared}");
            Console.Write(Environment.NewLine);
        }
    }
}
