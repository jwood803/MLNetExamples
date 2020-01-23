using Microsoft.Bot.Builder;
using Microsoft.Bot.Builder.Dialogs;
using Microsoft.Bot.Builder.Dialogs.Choices;
using Microsoft.Extensions.ML;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace PredictionBot.Dialogs
{
    public class HousingPredictionDialog : ComponentDialog
    {
        private readonly IStatePropertyAccessor<HousingState> _wineStateAccessor;
        private readonly PredictionEnginePool<HousingData, HousingPrediction> _predictionEnginePool;

        public HousingPredictionDialog(UserState userState, PredictionEnginePool<HousingData, HousingPrediction> predictionEnginePool) : base(nameof(HousingPredictionDialog))
        {
            _wineStateAccessor = userState.CreateProperty<HousingState>("WineState");

            _predictionEnginePool = predictionEnginePool;

            var steps = new WaterfallStep[]
            {
                LatitudeStepAsync,
                LongitudeStepAsync,
                HousingMedianAgeStepAsync,
                TotalRoomsStepAsync,
                FinishDialogAsync
            };

            AddDialog(new WaterfallDialog(nameof(WaterfallDialog), steps));
            AddDialog(new ChoicePrompt(nameof(ChoicePrompt)));
            AddDialog(new TextPrompt(nameof(TextPrompt)));
            AddDialog(new NumberPrompt<int>(nameof(NumberPrompt<int>)));

            InitialDialogId = nameof(WaterfallDialog);
        }

        private static async Task<DialogTurnResult> LatitudeStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            return await stepContext.PromptAsync(nameof(NumberPrompt<int>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the latitude?")
            });
        }

        private static async Task<DialogTurnResult> LongitudeStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["Latitude"] = ((FoundChoice)stepContext.Result).Value;

            return await stepContext.PromptAsync(nameof(NumberPrompt<int>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the longitude?")
            });
        }

        private static async Task<DialogTurnResult> HousingMedianAgeStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["Longitude"] = ((FoundChoice)stepContext.Result).Value;

            return await stepContext.PromptAsync(nameof(NumberPrompt<int>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the housing median age?")
            });
        }

        private static async Task<DialogTurnResult> TotalRoomsStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["MedianAge"] = ((FoundChoice)stepContext.Result).Value;

            return await stepContext.PromptAsync(nameof(NumberPrompt<int>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the number of total rooms?")
            });
        }

        private static async Task<DialogTurnResult> TotalBedroomsStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["TotalRooms"] = ((FoundChoice)stepContext.Result).Value;

            return await stepContext.PromptAsync(nameof(NumberPrompt<int>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the number of total bedrooms?")
            });
        }

        private static async Task<DialogTurnResult> PopulationStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["TotalBedrooms"] = ((FoundChoice)stepContext.Result).Value;

            return await stepContext.PromptAsync(nameof(NumberPrompt<int>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the population of the area?")
            });
        }

        private static async Task<DialogTurnResult> HouseholdsStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["Population"] = ((FoundChoice)stepContext.Result).Value;

            return await stepContext.PromptAsync(nameof(NumberPrompt<int>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the number of total bedrooms?")
            });
        }

        private static async Task<DialogTurnResult> FinishDialogAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {


            var endMessage = MessageFactory.Text("Predicted house price is...");

            await stepContext.Context.SendActivityAsync(endMessage, cancellationToken);

            return await stepContext.EndDialogAsync(cancellationToken: cancellationToken);
        }
    }
}
