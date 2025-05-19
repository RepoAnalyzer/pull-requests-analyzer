Сообщение: Related to https://github.com/tesseract-ocr/tesseract/issues/2990 (although it does not replace the exit() calls – yet).
@kba will provide an easily reproducable example of such failing assertions. (My current case is more special / harder to reconstruct.)
But regardless of the concrete issues that cause these failed assertions, which of course need to be addressed themselves, this makes Tesseract usable productively as a library. (You can catch these exceptions.)
(No idea what's the matter with CI here, btw.)
Сообщение: use std::filesystem::path instead of std::string for datadir, add warning if datadir is not directory or does not exists
Сообщение: Init time option lstm_num_threads should be used to set the number of LSTM threads. This will ensure that word recognition can run independently in multiple threads, thus effectively utilizing multi-core processors.
Following are my test results for a sample screenshot. CPU : Intel(R) Core(TM) i5-7500 CPU @ 3.40GHz OS : WIndows Compiler : MSVC 19.38.33130.0 (Installed from Visual Studio 2022) Model: eng.traineddata from tessfast PSM: 6
Total time taken for Recognize API call, Built without OpenMP With lstm_num_threads=1, total time taken = 3.95 seconds With lstm_num_threads=4, total time taken = 1.4 seconds
On the other hand, here are the numbers with OpenMP OMP_THREAD_LIMIT not set, total time taken = 3.59 seconds OMP_THREAD_LIMIT=4, total time taken = 3.57 seconds OMP_THREAD_LIMIT=1, total time taken = 4.19 seconds
As we can observe, this branch with lstm_num_threads set as 4, performs way better than the openmp multithreading supported currently. Setting lstm_num_threads equal to the number of cores in the processor will give the best performance.
