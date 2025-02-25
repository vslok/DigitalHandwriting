using DigitalHandwriting.Helpers;
using DigitalHandwriting.Services;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Windows.Input;
using System.Linq;
using DigitalHandwriting.Factories.AuthenticationMethods.Models;
using static DigitalHandwriting.Helpers.Calculations.BiometricMetrics;

namespace DigitalHandwriting.ViewModels
{
    public class AlgorithmValidationViewModel : BaseViewModel
    {
        private readonly AlgorithmValidationService _algorithmValidationService;
        private readonly DataMigrationService _dataMigrationService;
        private List<CsvExportAuthentication> _allResults;
        private List<CsvExportAuthentication> _displayedResults;
        private int _pageSize = 100;
        private int _currentPage = 0;
        private double _eer;
        private double _eerThreshold;
        private Dictionary<double, ThresholdMetrics> _thresholdMetrics;

        public ICommand OnValidationResultButtonImportClickCommand { get; set; }
        public ICommand OnImportButtonClickCommand { get; set; }
        public ICommand ValidateDataCommand { get; private set; }
        public ICommand NextPageCommand { get; private set; }
        public ICommand PreviousPageCommand { get; private set; }

        public double EER
        {
            get => _eer;
            set
            {
                _eer = value;
                InvokeOnPropertyChangedEvent(nameof(EER));
            }
        }

        public double EERThreshold
        {
            get => _eerThreshold;
            set
            {
                _eerThreshold = value;
                InvokeOnPropertyChangedEvent(nameof(EERThreshold));
            }
        }

        public Dictionary<double, ThresholdMetrics> ThresholdMetrics
        {
            get => _thresholdMetrics;
            set
            {
                _thresholdMetrics = value;
                InvokeOnPropertyChangedEvent(nameof(ThresholdMetrics));
            }
        }

        public AlgorithmValidationViewModel()
        {
            _algorithmValidationService = new AlgorithmValidationService();
            _dataMigrationService = new DataMigrationService();
            OnValidationResultButtonImportClickCommand = new Command(OnValidationResultButtonImportClick);
            OnImportButtonClickCommand = new Command(OnImportButtonClick);
            ValidateDataCommand = new Command(OnValidateDataClick);
            NextPageCommand = new Command(NextPage, CanGoToNextPage);
            PreviousPageCommand = new Command(PreviousPage, CanGoToPreviousPage);

            _allResults = new List<CsvExportAuthentication>();
            _displayedResults = new List<CsvExportAuthentication>();
            _thresholdMetrics = new Dictionary<double, ThresholdMetrics>();
        }

        public List<CsvExportAuthentication> ValidationResults
        {
            get => _displayedResults;
            set
            {
                _allResults = value;
                UpdateDisplayedResults();
                InvokeOnPropertyChangedEvent(nameof(ValidationResults));
            }
        }

        private void UpdateDisplayedResults()
        {
            _displayedResults = _allResults
                .Skip(_currentPage * _pageSize)
                .Take(_pageSize)
                .ToList();

            InvokeOnPropertyChangedEvent(nameof(ValidationResults));
            (NextPageCommand as Command)?.RaiseCanExecuteChanged();
            (PreviousPageCommand as Command)?.RaiseCanExecuteChanged();
        }

        private void NextPage()
        {
            _currentPage++;
            UpdateDisplayedResults();
        }

        private void PreviousPage()
        {
            _currentPage--;
            UpdateDisplayedResults();
        }

        private bool CanGoToNextPage()
        {
            return (_currentPage + 1) * _pageSize < _allResults.Count;
        }

        private bool CanGoToPreviousPage()
        {
            return _currentPage > 0;
        }

        private void OnImportButtonClick()
        {
            // Configure open file dialog box
            var dialog = new Microsoft.Win32.OpenFileDialog();
            dialog.FileName = "users"; // Default file name
            dialog.DefaultExt = ".csv"; // Default file extension
            dialog.Filter = "CSV documents (.csv)|*.csv"; // Filter files by extension

            // Show open file dialog box
            bool? result = dialog.ShowDialog();

            // Process open file dialog box results
            if (result == true)
            {
                // Open document
                string filename = dialog.FileName;
                _dataMigrationService.ImportDataFromCsv(filename);
            }
        }

        private void OnValidationResultButtonImportClick()
        {
            var dialog = new Microsoft.Win32.OpenFileDialog();
            dialog.FileName = "";
            dialog.DefaultExt = ".csv";
            dialog.Filter = "CSV (.csv)|*.csv";

            if (dialog.ShowDialog() == true)
            {
                string filename = dialog.FileName;
                var results = _dataMigrationService.GetAuthenticationResultsFromCsv(filename).ToList();
                _currentPage = 0;
                ValidationResults = results;

                // Update metrics using Calculations class
                if (results.Any())
                {
                    var (eer, eerThreshold, thresholdMetrics) = Calculations.BiometricMetrics.CalculateMetrics(results);
                    EER = eer;
                    EERThreshold = eerThreshold;
                    ThresholdMetrics = thresholdMetrics;
                }
            }
        }

        private string GetSaveDirectoryPath()
        {
            var dialog = new Microsoft.Win32.SaveFileDialog
            {
                Title = "Select folder to save validation results",
                FileName = "Select Folder", // Placeholder name
                CheckFileExists = false,
                CheckPathExists = true,
                ValidateNames = false
            };

            if (dialog.ShowDialog() == true)
            {
                return Path.GetDirectoryName(dialog.FileName);
            }

            return null;
        }

        private void OnValidateDataClick()
        {
            var dialog = new Microsoft.Win32.OpenFileDialog
            {
                FileName = "",
                DefaultExt = ".csv",
                Filter = "CSV (.csv)|*.csv",
                Title = "Select test data CSV file"
            };

            if (dialog.ShowDialog() != true) return;
            string testDataPath = dialog.FileName;

            var saveDirectory = GetSaveDirectoryPath();
            if (saveDirectory == null) return;

            try
            {
                // Run validation for N=1,2,3
                _algorithmValidationService.ValidateAuthentication(testDataPath, 1, saveDirectory);
                _algorithmValidationService.ValidateAuthentication(testDataPath, 2, saveDirectory);
                _algorithmValidationService.ValidateAuthentication(testDataPath, 3, saveDirectory);

                System.Windows.MessageBox.Show(
                    $"Validation completed successfully!\nResults saved in: {saveDirectory}",
                    "Validation Complete",
                    System.Windows.MessageBoxButton.OK,
                    System.Windows.MessageBoxImage.Information
                );
            }
            catch (Exception ex)
            {
                System.Windows.MessageBox.Show(
                    $"Error during validation: {ex.Message}",
                    "Validation Error",
                    System.Windows.MessageBoxButton.OK,
                    System.Windows.MessageBoxImage.Error
                );
            }
        }
    }
}
