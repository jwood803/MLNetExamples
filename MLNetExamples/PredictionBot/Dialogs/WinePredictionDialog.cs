using Microsoft.Bot.Builder;
using Microsoft.Bot.Builder.Dialogs;
using Microsoft.Bot.Builder.Dialogs.Choices;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace PredictionBot.Dialogs
{
    public class WinePredictionDialog : ComponentDialog
    {
        private readonly IStatePropertyAccessor<WineState> _wineStateAccessor;

        public WinePredictionDialog(UserState userState) : base(nameof(WinePredictionDialog))
        {
            _wineStateAccessor = userState.CreateProperty<WineState>("WineState");

            var steps = new WaterfallStep[]
            {
                WineTypeStepAsync,
                FixedAcidityStepAsync,
                FinishDialogAsync
            };

            AddDialog(new WaterfallDialog(nameof(WaterfallDialog), steps));
            AddDialog(new ChoicePrompt(nameof(ChoicePrompt)));
            AddDialog(new TextPrompt(nameof(TextPrompt)));
            AddDialog(new NumberPrompt<int>(nameof(NumberPrompt<int>)));

            InitialDialogId = nameof(WaterfallDialog);
        }

        private static async Task<DialogTurnResult> WineTypeStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            return await stepContext.PromptAsync(nameof(ChoicePrompt),
                new PromptOptions
                {
                    Prompt = MessageFactory.Text("Red or white wine?"),
                    Choices = ChoiceFactory.ToChoices(new List<string> { "Red", "White" })
                }, cancellationToken);
        }

        private static async Task<DialogTurnResult> FixedAcidityStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["wineType"] = ((FoundChoice)stepContext.Result).Value;

            return await stepContext.PromptAsync(nameof(TextPrompt), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the fixed acidity?")
            });
        }

        private static async Task<DialogTurnResult> FinishDialogAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            var endMessage = MessageFactory.Text("Predicted wine quality is...");

            await stepContext.Context.SendActivityAsync(endMessage, cancellationToken);

            return await stepContext.EndDialogAsync(cancellationToken: cancellationToken);
        }
    }
}
