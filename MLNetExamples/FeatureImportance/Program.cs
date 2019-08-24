using Microsoft.ML;
using System;
using System.Linq;

namespace FeatureImportance
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<HousingData>("./housing.csv", hasHeader: true, separatorChar: ',');

            var preview = data.Preview();

            var features = data.Schema
                .Select(col => col.Name)
                .Where(colName => colName != "Label" && colName != "OceanProximity")
                .ToArray();

            var pipeline = context.Transforms.Text.FeaturizeText("Text", "OceanProximity")
                .Append(context.Transforms.Concatenate("Features", features))
                .Append(context.Regression.Trainers.LbfgsPoissonRegression());

            var model = pipeline.Fit(data);
            var transformedData = model.Transform(data);

            // Get weights of model
            var linearModel = model.LastTransformer.Model;

            var weights = linearModel.Weights;

            Console.WriteLine("Weights:");
            var weightsResult = "";

            foreach (var weight in weights)
            {
                weightsResult += $"{weight} ";
            }

            Console.WriteLine(weightsResult);
            Console.WriteLine(Environment.NewLine);

            // Get global feature importance
            var lastTransformer = model.LastTransformer;

            var featureImportance = context.Regression.PermutationFeatureImportance(lastTransformer, transformedData);

            Console.WriteLine("Global feature importance:");
            for (int i = 0; i < featureImportance.Count(); i++)
            {
                Console.WriteLine($"Feature - {features[i]}: Difference in RMS - {featureImportance[i].RootMeanSquaredError.Mean}");
            }

            Console.WriteLine(Environment.NewLine);

            // Get feature importance for each row
            var featureContribution = context.Transforms.CalculateFeatureContribution(lastTransformer, normalize: false);

            var featureContributionResults = featureContribution.Fit(transformedData).Transform(transformedData);

            var topTenRows = context.Data.TakeRows(featureContributionResults, 10);

            var scoringEnumerator = context.Data.CreateEnumerable<HousingData>(topTenRows, reuseRowObject: true);

            Console.WriteLine("Row feature importance:");
            Console.WriteLine("Households - Housing Median Age - Median Income - Ocean Proximity");
            var globalResults = "";
            foreach (var row in scoringEnumerator)
            {
                globalResults += $"{row.Households} - {row.HousingMedianAge} = {row.MedianIncome} - {row.OceanProximity}\n";
            }

            Console.WriteLine(globalResults);

            Console.ReadLine();
        }
    }
}
