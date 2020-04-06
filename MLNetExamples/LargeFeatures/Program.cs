using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;

namespace LargeFeatures
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<SensorData>("./features.csv", hasHeader: false, separatorChar: ',');

            var pipeline = context.Transforms.Conversion.MapValueToKey("Label", "Gas")
                .Append(context.MulticlassClassification.Trainers.LbfgsMaximumEntropy());

            var model = pipeline.Fit(data);

            var predictions = model.Transform(data);

            var metrics = context.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"Log loss - {metrics.LogLoss}");

            Console.ReadLine();
        }
    }
}
