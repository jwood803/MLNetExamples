using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Security.Principal;

namespace NormalizeText
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var emptyData = context.Data.LoadFromEnumerable(new List<Input>());

            var normalizedPipeline = context.Transforms.Text.NormalizeText("NormalizedText", "Text",    
                Microsoft.ML.Transforms.Text.TextNormalizingEstimator.CaseMode.Upper, keepDiacritics: false, keepPunctuations: false, keepNumbers: true);

            var normalizeTransformer = normalizedPipeline.Fit(emptyData);

            var predictionEngine = context.Model.CreatePredictionEngine<Input, Output>(normalizeTransformer);

            var text = new Input { Text = "Whisk the batter for the crêpe, then let it sit for 5 minutes." };

            var normalizedText = predictionEngine.Predict(text);

            Console.WriteLine($"Original text - {text.Text}");
            Console.WriteLine(Environment.NewLine);
            Console.WriteLine($"Normalized text - {normalizedText.NormalizedText}");
        }
    }
}
