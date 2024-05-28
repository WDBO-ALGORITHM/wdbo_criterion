typedef enum
{
    ERR_CLANG_TYPE_FIX = -1, // this stupid value is to fix type to be int instead of unsigned on some compilers (e.g. clang version 8.0)
    ERR_NONE = 0,            // no error

    ERR_MATRIX_NOT_PSD,
    ERR_MATRIX_NOT_INVERTIBLE,
    ERR_NEGATIVE_SQRT,
    ERR_NB_ERR // not an actual error but to have the total number of errors
} error_code;