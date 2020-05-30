using Microsoft.ML.Data;

namespace Ranking
{
    internal class RankingPrediction
    {
        [ColumnName("Score")]
        public float RelevanceScore { get; set; }
    }
}