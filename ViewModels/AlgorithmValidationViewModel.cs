using DigitalHandwriting.Services;
using System.Collections.Generic;

namespace DigitalHandwriting.ViewModels
{
    public class AlgorithmValidationViewModel : BaseViewModel
    {
        private readonly AlgorithmValidationService _algorithmValidationService;
        private List<AuthenticationValidationResult> _results;

        public AlgorithmValidationViewModel(string testDataPath)
        {
            _algorithmValidationService = new AlgorithmValidationService();
            ValidationResults = _algorithmValidationService.ValidateAuthentication(testDataPath);
        }

        public List<AuthenticationValidationResult> ValidationResults {
            set => SetProperty(ref _results, value);
            get => _results;
        }
    }
}
