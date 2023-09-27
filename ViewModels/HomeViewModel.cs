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
        public ICommand OnAuthentificationButtonClickCommand { get; set; }
        public ICommand OnCheckTextBoxKeyDownEventCommand { get; set; }
        public ICommand OnCheckTextBoxKeyUpEventCommand { get; set; }

        private int _authentificationTry = 0;

        private int _checkTextCurrentLetterIndex = 0;

        private bool _isAuthentificationButtonEnabled = false;

        private bool _isPasswordAuthentificationEnabled = false;

        private bool _isHandwritingAuthentificationEnabled = true;

        private string _checkTextWithUpperCase => ConstStrings.CheckText.ToUpper();

        private string _userCheckText = "";

        private string _userLogin = "";

        private string _userPassword = "";

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

            OnAuthentificationButtonClickCommand = new Command(OnAuthentificationButtonClick);
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

        public bool IsPasswordAuthentificationEnabled
        {
            get => _isPasswordAuthentificationEnabled;
            set => SetProperty(ref _isPasswordAuthentificationEnabled, value);
        }

        public bool IsAuthentificationButtonEnabled
        {
            get => _isAuthentificationButtonEnabled;
            set => SetProperty(ref _isAuthentificationButtonEnabled, value);
        }

        public string UserLogin
        {
            get => _userLogin;
            set
            {
                _authentificationTry = 0;
                SetProperty(ref _userLogin, value);
            }
        }

        public string UserPassword
        {
            get => _userPassword; 
            set
            {
                SetProperty(ref _userPassword, value);
                if (_userPassword.Length > 8)
                {
                    IsAuthentificationButtonEnabled = true;
                }
            }
        }

        public string UserCheckText
        {
            get => _userCheckText;
            set => SetProperty(ref _userCheckText, value);
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
                else
                {
                    IsAuthentificationButtonEnabled = true;
                }
            }
        }

        private void OnCheckTextBoxKeyDownEvent(object props)
        {
            var e = (KeyEventArgs)props;
            if (!Helper.CheckCurrentLetterKeyPressed(e, _checkTextCurrentLetterIndex, _checkTextWithUpperCase))
            {
                e.Handled = true;
                return;
            }

            _checkTextCurrentLetterIndex++;
            _keyboardMetricsCollector.OnKeyDownEvent(e);
        }

        private void OnAuthentificationButtonClick()
        {
            var user = db.Users.Select(user => user).Where(user => user.Login == UserLogin).FirstOrDefault();

            if (IsPasswordAuthentificationEnabled)
            {
                var IsAuthentificated = AuthentificationService.PasswordAuthentification(user, UserPassword);
                if (!IsAuthentificated)
                {
                    ResetWindowState();
                    return;
                }
            }

            if (_authentificationTry <= 2)
            {
                _keyboardMetricsCollector.GetCurrentStepValues(_checkTextWithUpperCase, out var keyPressedValues, out var betweenKeysValues);

                var isAuthentificated = AuthentificationService.HandwritingAuthentification(user, keyPressedValues, betweenKeysValues,
                    out double keyPressedDistance, out double beetweenKeysDistance);

                var window = new UserInfo(user, keyPressedValues, betweenKeysValues);
                window.ShowDialog();

                if (_authentificationTry == 2)
                {
                    EnablePasswordAuthentification();
                }

                if (!isAuthentificated)
                {
                    _authentificationTry++;
                }
            }

            ResetTryState();
        }

        private void ResetTryState()
        {
            IsAuthentificationButtonEnabled = false;
            _checkTextCurrentLetterIndex = 0;
            UserCheckText = "";
            UserPassword = "";
            _keyboardMetricsCollector.ResetMetricsCollection();
        }

        private void ResetWindowState()
        {
            ResetTryState();
            IsPasswordAuthentificationEnabled = false;
            IsHandwritingAuthentificationEnabled = true;
            UserLogin = "";
        }

        private void EnablePasswordAuthentification()
        {
            UserCheckText = "";
            _checkTextCurrentLetterIndex = 0;
            IsPasswordAuthentificationEnabled = true;
            IsHandwritingAuthentificationEnabled = false;
        }
    }
}
