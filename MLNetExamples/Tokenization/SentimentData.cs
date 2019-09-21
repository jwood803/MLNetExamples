using Microsoft.ML.Data;

namespace Tokenization
{
    public class SentimentData
    {
        [LoadColumn(0)]
        public bool Sentiment { get; set; }

        [LoadColumn(1)]
        public string Text { get; set; }
    }
}
