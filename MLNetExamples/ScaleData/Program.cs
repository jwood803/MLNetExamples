using Microsoft.ML;
using System;
using System.Linq;

namespace ScaleData
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<HousingData>("./housing.csv", hasHeader: true, separatorChar: ',');

            var columns = data.Schema
                .Select(col => col.Name)
                .Where(col => col != "Label" && col != "OceanProximity")
                .Select(col => new InputOutputColumnPair(col, col))
                .ToArray();

            var scaling = context.Transforms.NormalizeMinMax(columns);

            var dataScaled = scaling.Fit(data).Transform(data);

            var features = columns.Select(col => col.OutputColumnName).ToArray();

            var preview = dataScaled.Preview();

            var pipeline = context.Transforms.Text.FeaturizeText("Text", "OceanProximity")
                .Append(context.Transforms.Concatenate("Features", features))
                .Append(context.Transforms.Concatenate("Features", "Text"))
                .Append(context.Regression.Trainers.LbfgsPoissonRegression());

            var model = pipeline.Fit(dataScaled);

            var predictionFunc = context.Model.CreatePredictionEngine<HousingData, HousingPrediction>(model);

            var prediction = predictionFunc.Predict(new HousingData
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
            });

            Console.WriteLine($"Prediction - {prediction.PredictedHouseValue}");

            Console.ReadLine();
        }
    }
}
