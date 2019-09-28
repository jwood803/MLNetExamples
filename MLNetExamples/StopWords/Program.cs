using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;

namespace StopWords
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var emptyData = new List<TextData>();

            var data = context.Data.LoadFromEnumerable(emptyData);

            var tokenization = context.Transforms.Text.TokenizeIntoWords("Tokens", "Text", separators: new[] { ' ', '.', ',' })
                .Append(context.Transforms.Text.RemoveDefaultStopWords("Tokens", "Tokens",
                    Microsoft.ML.Transforms.Text.StopWordsRemovingEstimator.Language.English));

            var stopWordsModel = tokenization.Fit(data);

            var engine = context.Model.CreatePredictionEngine<TextData, TextTokens>(stopWordsModel);

            var newText = engine.Predict(new TextData { Text = "This is a test sentence, and it is a long one." });

            PrintTokens(newText);

            var customTokenization = context.Transforms.Text.TokenizeIntoWords("Tokens", "Text", separators: new[] { ' ', '.', ',' })
                .Append(context.Transforms.Text.RemoveStopWords("Tokens", "Tokens", new[] { "and", "a" }));

            var customStopWordsModel = customTokenization.Fit(data);

            var customEngine = context.Model.CreatePredictionEngine<TextData, TextTokens>(customStopWordsModel);

            var newCustomText = customEngine.Predict(new TextData { Text = "This is a test sentence, and it is a long one." });

            PrintTokens(newCustomText);

            Console.ReadLine();
        }

        private static void PrintTokens(TextTokens tokens)
        {
            Console.WriteLine(Environment.NewLine);

            var sb = new StringBuilder();

            foreach (var token in tokens.Tokens)
            {
                sb.AppendLine(token);
            }

            Console.WriteLine(sb.ToString());
        }
    }
}
