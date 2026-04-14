package ai.ondev.snapdragonlab.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

private val AppTypography = Typography(
    displayLarge = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.Bold, fontSize = 57.sp),
    headlineLarge = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.SemiBold, fontSize = 32.sp),
    headlineMedium = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.SemiBold, fontSize = 28.sp),
    headlineSmall = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.SemiBold, fontSize = 24.sp),
    titleLarge = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.SemiBold, fontSize = 22.sp),
    titleMedium = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.Medium, fontSize = 16.sp),
    titleSmall = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.Medium, fontSize = 14.sp),
    bodyLarge = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.Normal, fontSize = 16.sp),
    bodyMedium = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.Normal, fontSize = 14.sp),
    bodySmall = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.Normal, fontSize = 12.sp),
    labelLarge = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.Medium, fontSize = 14.sp),
    labelMedium = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.Medium, fontSize = 12.sp),
    labelSmall = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.Medium, fontSize = 11.sp),
)

// Deep indigo/purple dark palette
private val DarkColorScheme = darkColorScheme(
    primary = Color(0xFF9FA8DA),        // Soft indigo
    onPrimary = Color(0xFF1A1B3A),
    primaryContainer = Color(0xFF3949AB),
    onPrimaryContainer = Color(0xFFDDE0FF),
    secondary = Color(0xFF80CBC4),       // Teal accent
    onSecondary = Color(0xFF003733),
    secondaryContainer = Color(0xFF00695C),
    onSecondaryContainer = Color(0xFFA7F3EC),
    tertiary = Color(0xFFCE93D8),        // Soft purple
    onTertiary = Color(0xFF3B0A43),
    tertiaryContainer = Color(0xFF6A1B9A),
    onTertiaryContainer = Color(0xFFF3D9F8),
    error = Color(0xFFEF9A9A),
    onError = Color(0xFF601410),
    background = Color(0xFF0D0E1A),      // Very dark navy
    onBackground = Color(0xFFE3E3F0),
    surface = Color(0xFF151627),          // Slightly lighter navy
    onSurface = Color(0xFFE3E3F0),
    surfaceVariant = Color(0xFF1E2036),   // Card surface
    onSurfaceVariant = Color(0xFFC5C6D9),
    outline = Color(0xFF7E7F96),
    outlineVariant = Color(0xFF44455E),
    surfaceContainerLowest = Color(0xFF0A0B15),
    surfaceContainerLow = Color(0xFF121321),
    surfaceContainer = Color(0xFF171829),
    surfaceContainerHigh = Color(0xFF1C1D31),
    surfaceContainerHighest = Color(0xFF22233B),
)

private val LightColorScheme = lightColorScheme(
    primary = Color(0xFF3949AB),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFDDE0FF),
    onPrimaryContainer = Color(0xFF0D1163),
    secondary = Color(0xFF00796B),
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFA7F3EC),
    onSecondaryContainer = Color(0xFF002019),
    tertiary = Color(0xFF7B1FA2),
    onTertiary = Color.White,
    tertiaryContainer = Color(0xFFF3D9F8),
    onTertiaryContainer = Color(0xFF2C0036),
    background = Color(0xFFF8F8FC),
    onBackground = Color(0xFF1B1B1F),
    surface = Color(0xFFFFFFFF),
    onSurface = Color(0xFF1B1B1F),
    surfaceVariant = Color(0xFFE7E8F0),
    onSurfaceVariant = Color(0xFF44455E),
)

@Composable
fun OnDevAITheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit,
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = AppTypography,
        content = content,
    )
}
