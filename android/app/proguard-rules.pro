# Keep rules for the release build (R8 with resource shrinking).

# --- Gym page bridge -------------------------------------------------------------------
# Methods the gym page calls through addJavascriptInterface are looked up by name.
-keepattributes RuntimeVisibleAnnotations,AnnotationDefault,InnerClasses,Signature
-keepclassmembers class * {
    @android.webkit.JavascriptInterface <methods>;
}

# --- kotlinx.serialization -------------------------------------------------------------
# The library ships its own consumer rules; these cover @Serializable classes with named
# companions and serializable objects, as recommended in its README.
-if @kotlinx.serialization.Serializable class **
-keepclassmembers class <1> {
    static <1>$* Companion;
}
-if @kotlinx.serialization.Serializable class ** {
    static **$* *;
}
-keepclassmembers class <2>$<3> {
    kotlinx.serialization.KSerializer serializer(...);
}
-if @kotlinx.serialization.Serializable class ** {
    public static ** INSTANCE;
}
-keepclassmembers class <1> {
    public static <1> INSTANCE;
    kotlinx.serialization.KSerializer serializer(...);
}

# --- MediaPipe Tasks -------------------------------------------------------------------
# The SDK reaches its Java classes from native code and reads protobuf-lite messages by
# reflection on field names.
-keep class com.google.mediapipe.** { *; }
-keep class com.google.protobuf.** { *; }
-keepclassmembers class * extends com.google.protobuf.GeneratedMessageLite {
    <fields>;
}
# MediaPipe logs through Flogger, which finds the calling class by walking the stack and
# comparing class names. Renamed or inlined, it throws "no caller found on the stack" from
# Graph's static initializer and the first analysis crashes.
-keep class com.google.common.flogger.** { *; }
-dontwarn com.google.mediapipe.proto.CalculatorProfileProto$CalculatorProfile
-dontwarn com.google.mediapipe.proto.GraphTemplateProto$CalculatorGraphTemplate
-dontwarn javax.lang.model.**
-dontwarn javax.annotation.processing.**
-dontwarn com.google.auto.value.**
-dontwarn org.checkerframework.**
-dontwarn com.google.errorprone.annotations.**
