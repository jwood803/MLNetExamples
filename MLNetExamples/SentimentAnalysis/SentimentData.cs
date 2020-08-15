using Microsoft.ML.Data;

namespace SentimentAnalysis
{
    public class SentimentData
    {
        [LoadColumn(0)]
        public string Text { get; set; }

        [LoadColumn(1)]
        public float Sentiment { get; set; }
    }
}
