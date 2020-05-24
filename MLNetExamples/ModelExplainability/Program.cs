using Microsoft.ML;
using System;
using System.Linq;

namespace ModelExplainability
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<HousingData>("./housing.csv", hasHeader: true, separatorChar: ',');

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

            for (int i = 0; i < features.Length; i++)
            {
                Console.WriteLine($"Feature {features[i]} has weight {weights[i]}");
            }

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
            var firstRow = model.Transform(context.Data.TakeRows(transformedData, 1));

            var featureContribution = context.Transforms.CalculateFeatureContribution(lastTransformer, normalize: false);

            var featureContributionTransformer = featureContribution.Fit(firstRow);

            var featureContributionPipeline = model.Append(featureContributionTransformer);

            var predictionEngine = context.Model.CreatePredictionEngine<HousingData, HousingPrediction>(featureContributionPipeline);

            var sampleData = new HousingData
            {
                Longitude = -122.25f,
                Latitude = 37.85f,
                HousingMedianAge = 55.0f,
                TotalRooms = 1627.0f,
                TotalBedrooms = 235.0f,
                Population = 322.0f,
                Households = 120.0f,
                MedianIncome = 8.3014f,
                OceanProximity = "NEAR BAY"
            };

            var prediction = predictionEngine.Predict(sampleData);

            Console.WriteLine("Row feature importance:");

            for (int i = 0; i < prediction.FeatureContributions.Length; i++)
            {
                Console.WriteLine($"Feature {features[i]} has feature contribution of {prediction.FeatureContributions[i]}");
            }
        }
    }
}
