using DigitalHandwriting.Context;
using DigitalHandwriting.Helpers;
using DigitalHandwriting.Stores;
using DigitalHandwriting.Views;
using Microsoft.EntityFrameworkCore;
using System;
using System.Linq;
using System.Windows.Input;
using DigitalHandwriting.Commands;
using DigitalHandwriting.Services;
using System.Windows.Documents;
using DigitalHandwriting.Repositories;
using Microsoft.Win32;

namespace DigitalHandwriting.ViewModels
{
    public class HomeViewModel : BaseViewModel
    {
        public ICommand OnRegistrationButtonClickCommand { get; set; }
        public ICommand OnAuthenticationButtonClickCommand { get; set; }
        public ICommand OnCheckTextBoxKeyDownEventCommand { get; set; }
        public ICommand OnCheckTextBoxKeyUpEventCommand { get; set; }
        public ICommand OnImportButtonClickCommand { get; set; }
        public ICommand OnValidationButtonClickCommand { get; set; }

        private int _authentificationTry = 0;

        private int _checkTextCurrentLetterIndex = 0;

        private bool _isHandwritingAuthentificationEnabled = true;

        private string _checkTextWithUpperCase => ConstStrings.CheckText.ToUpper();

        private string _userCheckText = "";

        private string _userLogin = "";

        private readonly UserRepository _userRepository;

        private readonly KeyboardMetricsCollectionService _keyboardMetricsCollector;

        private readonly DataMigrationService _dataMigrationService;

        public HomeViewModel(
            Func<RegistrationViewModel> registrationViewModelFactory,
            NavigationStore navigationStore,
            KeyboardMetricsCollectionService keyboardMetricsCollector)
        {
            OnRegistrationButtonClickCommand = new NavigateCommand<RegistrationViewModel>(
                new NavigationService<RegistrationViewModel>(
                    navigationStore,
                    () => registrationViewModelFactory()));

            OnAuthenticationButtonClickCommand = new Command(OnAuthenticationButtonClick);
            OnImportButtonClickCommand = new Command(OnImportButtonClick);
            OnValidationButtonClickCommand = new Command(OnValidationButtonClick);
            OnCheckTextBoxKeyDownEventCommand = new RelayCommand<object>(OnCheckTextBoxKeyDownEvent);
            OnCheckTextBoxKeyUpEventCommand = new RelayCommand<object>(OnCheckTextBoxKeyUpEvent);

            _keyboardMetricsCollector = keyboardMetricsCollector;
            _userRepository = new UserRepository();
            _dataMigrationService = new DataMigrationService();
        }

        public bool IsHandwritingAuthentificationEnabled
        {
            get => _isHandwritingAuthentificationEnabled;
            set => SetProperty(ref _isHandwritingAuthentificationEnabled, value);
        }

        public bool IsAuthenticationButtonEnabled => UserLogin.Length > 0 && UserCheckText.Length >= 20;

        public string UserLogin
        {
            get => _userLogin;
            set
            {
                _authentificationTry = 0;
                SetProperty(ref _userLogin, value);
                InvokeOnPropertyChangedEvent(nameof(IsAuthenticationButtonEnabled));
            }
        }

        public string UserCheckText
        {
            get => _userCheckText;
            set
            {
                SetProperty(ref _userCheckText, value);
                InvokeOnPropertyChangedEvent(nameof(IsAuthenticationButtonEnabled));
            }
        }

        private void OnCheckTextBoxKeyUpEvent(object props)
        {
            var e = (KeyEventArgs)props;

            _keyboardMetricsCollector.OnKeyUpEvent(e);

            if (_checkTextCurrentLetterIndex == _checkTextWithUpperCase.Length && UserCheckText.Length == _checkTextWithUpperCase.Length)
            {
                var user = _userRepository.GetUser(UserLogin);
                if (user == null)
                {
                    ResetTryState();
                }
            }
        }

        private void OnCheckTextBoxKeyDownEvent(object props)
        {
            var e = (KeyEventArgs)props;

            _keyboardMetricsCollector.OnKeyDownEvent(e);
        }

        private void OnAuthenticationButtonClick()
        {
            var user = _userRepository.GetUser(UserLogin);

            if (user == null)
            {
                // TODO: Add UI notification
                ResetTryState();
                return;
            }

            if (_authentificationTry <= 2)
            {
                var isPassPhraseValid = AuthenticationService.PasswordAuthentication(user, UserCheckText);
                if (!isPassPhraseValid)
                {
                    // TODO: Add UI notification
                    ResetTryState();
                    return;
                }

                _keyboardMetricsCollector.GetCurrentStepValues(
                    UserCheckText.ToUpper(), 
                    out var keyPressedValues, 
                    out var betweenKeysValues);

                var authenticationResult = AuthenticationService.HandwritingAuthentication(user, keyPressedValues, betweenKeysValues);

                var window = new UserInfo(authenticationResult);
                window.ShowDialog();

                if (!authenticationResult.IsAuthenticated)
                {
                    _authentificationTry++;
                }
            }

            ResetTryState();
        }

        private void OnImportButtonClick()
        {
            // Configure open file dialog box
            var dialog = new OpenFileDialog();
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

        private void OnValidationButtonClick()
        {
            var window = new ValidationResult();
            window.ShowDialog();
        }

        private void ResetTryState()
        {
            _checkTextCurrentLetterIndex = 0;
            UserCheckText = "";
            _keyboardMetricsCollector.ResetMetricsCollection();
        }

        private void ResetWindowState()
        {
            ResetTryState();
            IsHandwritingAuthentificationEnabled = true;
            UserLogin = "";
        }
    }
}
