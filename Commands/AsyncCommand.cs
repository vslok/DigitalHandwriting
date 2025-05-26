using System;
using System.Threading.Tasks;
using System.Windows.Input;

namespace DigitalHandwriting.Commands
{
    public class AsyncCommand : ICommand
    {
        private readonly Func<Task> _execute;
        private readonly Func<bool> _canExecute;
        private bool _isExecuting;

        public event EventHandler CanExecuteChanged;

        public AsyncCommand(Func<Task> execute, Func<bool> canExecute = null)
        {
            _execute = execute ?? throw new ArgumentNullException(nameof(execute));
            _canExecute = canExecute;
        }

        public bool CanExecute(object parameter)
        {
            return !_isExecuting && (_canExecute?.Invoke() ?? true);
        }

        public async void Execute(object parameter)
        {
            if (IsExecuting) return; // Check before setting IsExecuting to true

            IsExecuting = true;
            try
            {
                await _execute();
            }
            finally
            {
                IsExecuting = false;
            }
        }

        public void RaiseCanExecuteChanged()
        {
            CanExecuteChanged?.Invoke(this, EventArgs.Empty);
        }

        // Property to manage execution state and notify CanExecuteChanged
        private bool IsExecuting
        {
            get => _isExecuting;
            set
            {
                if (_isExecuting == value) return;
                _isExecuting = value;
                // Ensure UI updates on the main thread if CanExecuteChanged is raised from a background task
                // For this specific AsyncCommand, Execute is async void but awaited task runs on UI thread or worker.
                // If _execute() itself spawns UI work, that work should be dispatched.
                // CanExecuteChanged should be fine as it's usually checked by UI elements on the UI thread.
                RaiseCanExecuteChanged();
            }
        }
    }
}
