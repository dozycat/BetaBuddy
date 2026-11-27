package com.betabuddy.android.i18n

import com.betabuddy.android.StringResources
import org.junit.Test

class LibraryStringsTest {
    @Test fun permissionsProgressAndCorrectionsAreTranslated() {
        StringResources.assertComplete("strings_library.xml",atLeast = 16)
        StringResources.assertNoChineseLeftovers("strings_library.xml")
    }
}
