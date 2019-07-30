using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;

namespace AnomalyDetection
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<EnergyData>("./energy_hourly.csv", 
                hasHeader: true, 
                separatorChar: ',');

            var pipeline = context.Transforms.DetectSpikeBySsa(nameof(EnergyPrediction.Prediction), nameof(EnergyData.Energy),
                confidence: 98, trainingWindowSize: 90, seasonalityWindowSize: 30, pvalueHistoryLength: 30);

            var transformedData = pipeline.Fit(data).Transform(data);

            var predictions = context.Data.CreateEnumerable<EnergyPrediction>(transformedData, reuseRowObject: false).ToList();

            var energy = data.GetColumn<float>("Energy").ToArray();
            var date = data.GetColumn<DateTime>("Date").ToArray();

            Console.WriteLine("Anomalies:");
            for (int i = 0; i < predictions.Count(); i++)
            {
                if (predictions[i].Prediction[0] == 1)
                {
                    Console.WriteLine("{0}\t{1:0.0000}\t{2:0.00}\t{3:0.00}\t{4:0.00}",
                        date[i], energy[i], predictions[i].Prediction[0], predictions[i].Prediction[1], predictions[i].Prediction[2]);
                }
            }

            Console.ReadLine();
        }
    }
}
