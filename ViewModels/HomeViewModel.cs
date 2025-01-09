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

namespace DigitalHandwriting.ViewModels
{
    public class HomeViewModel : BaseViewModel
    {
        public ICommand OnRegistrationButtonClickCommand { get; set; }
        public ICommand OnAuthenticationButtonClickCommand { get; set; }
        public ICommand OnCheckTextBoxKeyDownEventCommand { get; set; }
        public ICommand OnCheckTextBoxKeyUpEventCommand { get; set; }

        private int _authentificationTry = 0;

        private int _checkTextCurrentLetterIndex = 0;

        private bool _isHandwritingAuthentificationEnabled = true;

        private string _checkTextWithUpperCase => ConstStrings.CheckText.ToUpper();

        private string _userCheckText = "";

        private string _userLogin = "";

        private ApplicationContext db = new ApplicationContext();

        private readonly KeyboardMetricsCollectionService _keyboardMetricsCollector;

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
            OnCheckTextBoxKeyDownEventCommand = new RelayCommand<object>(OnCheckTextBoxKeyDownEvent);
            OnCheckTextBoxKeyUpEventCommand = new RelayCommand<object>(OnCheckTextBoxKeyUpEvent);

            db.Database.EnsureCreated();

            _keyboardMetricsCollector = keyboardMetricsCollector;
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
                db.Users.Load();

                var userRegistrated = db.Users.Local.Select(user => user.Login).Contains(_userLogin);
                if (!userRegistrated)
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
            var user = db.Users.Select(user => user).Where(user => user.Login == UserLogin).FirstOrDefault();

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

                _keyboardMetricsCollector.GetCurrentStepValues(UserCheckText.ToUpper(), out var keyPressedValues, out var betweenKeysValues);

                var isAuthentificated = AuthenticationService.HandwritingAuthentication(user, keyPressedValues, betweenKeysValues,
                    out double keyPressedDistance, out double beetweenKeysDistance);

                var window = new UserInfo(user, keyPressedValues, betweenKeysValues);
                window.ShowDialog();

                if (!isAuthentificated)
                {
                    _authentificationTry++;
                }
            }

            ResetTryState();
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
