using DigitalHandwriting.Helpers;
using DigitalHandwriting.Services;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Windows.Input;

namespace DigitalHandwriting.ViewModels
{
    public class AlgorithmValidationViewModel : BaseViewModel
    {
        private readonly AlgorithmValidationService _algorithmValidationService;
        private List<AuthenticationValidationResult> _results;

        public ICommand OnValidationResultButtonExportClickCommand { get; set; }

        public AlgorithmValidationViewModel(string testDataPath)
        {
            _algorithmValidationService = new AlgorithmValidationService();
            OnValidationResultButtonExportClickCommand = new Command(OnValidationResultButtonExportClick);

            ValidationResults = _algorithmValidationService.ValidateAuthentication(testDataPath);
        }

        public List<AuthenticationValidationResult> ValidationResults {
            set => SetProperty(ref _results, value);
            get => _results;
        }

        private void OnValidationResultButtonExportClick()
        {
            var dialog = new Microsoft.Win32.SaveFileDialog();
            dialog.FileName = $"{DateTime.Now.ToShortDateString()} - validationResult"; // Default file name
            dialog.DefaultExt = ".json"; // Default file extension
            dialog.Filter = "JSON (.json)|*.json"; // Filter files by extension

            Nullable<bool> result = dialog.ShowDialog();

            // Process save file dialog box results
            if (result == true)
            {
                // Save document
                string filename = dialog.FileName;
                var jsonResult = JsonSerializer.Serialize(ValidationResults);
                File.WriteAllText(filename, jsonResult);
            }
        }
    }
}
