package com.betabuddy.android.journal

import org.junit.Assert.*
import org.junit.Test

class LibraryPolicyTest {
    private val policy = LibraryClassifierConfig(1,"LibraryImageEncoder",5,3,.5,.2,
        listOf(LibraryClassifierConfig.Prototype("a",listOf(1.0,0.0,0.0)), LibraryClassifierConfig.Prototype("b",listOf(0.0,1.0,0.0)), LibraryClassifierConfig.Prototype(null,listOf(0.0,0.0,1.0))),
        LibraryClassifierConfig.Model("library-image-encoder.onnx","0".repeat(64)))
    @Test fun ambiguityInvalidInputsAndVoteTiesAreNotMatches() {
        policy.validate()
        assertEquals("a",policy.tag(listOf(.8,.1,.1)))
        assertNull(policy.tag(listOf(.55,.5,.1)))
        assertNull(policy.tag(listOf(.1,.2,.8)))
        assertNull(policy.tag(listOf(Double.NaN,.1,.1)))
        assertNull(policy.tag(listOf(.8,.1)))
        assertNull(policy.decision(listOf("a","b",null,"a","b")))
        assertEquals("a",policy.decision(listOf("a",null,"a","b","a")))
        assertThrows(IllegalArgumentException::class.java) { policy.copy(minimumVotes = 6).validate() }
        assertThrows(IllegalArgumentException::class.java) { policy.copy(androidModel = policy.androidModel.copy(file = "../model")).validate() }
        assertEquals("a",policy.frameVote(listOf("a","a")))
        assertEquals("b",policy.frameVote(listOf(null,"b")))
        assertNull(policy.frameVote(listOf("a","b")))
        assertNull(policy.decision(listOf(policy.frameVote(listOf("a","a")),null,null,null,null)))
        assertThrows(IllegalArgumentException::class.java) { policy.copy(schema=2).validate() }
        policy.copy(schema=2,imageResize="centerAndFit").validate()
        assertEquals(listOf("center","fit"),policy.copy(imageResize="centerAndFit").imageModes)
        assertThrows(IllegalArgumentException::class.java) { policy.copy(imageResize="unsupported").validate() }
    }
    @Test fun onlyIndoorVideoCoordinatesCanSuggestAGym() {
        val origin = LibraryMetadata.location("+31.1000+121.2000+5.0/")!!
        assertEquals(31.1,origin.first,0.00001)
        assertNull(LibraryMetadata.location("+91.0+121.0/")); assertNull(LibraryMetadata.location("0,0"))
        assertNull(LibraryMetadata.location("+00.0+000.0/"))
        val gyms = listOf(DirectoryGym("coarse","City centre",lat=31.1,lon=121.2,coordPrecision="city"), DirectoryGym("near","Near",lat=31.1001,lon=121.2001,coordPrecision="poi"))
        assertEquals("near",LibraryMetadata.suggestedGym("indoor",origin,gyms)?.id)
        assertNull(LibraryMetadata.suggestedGym("outdoor",origin,gyms))
        assertNull(LibraryMetadata.suggestedGym("indoor",null,gyms))
        assertNull(LibraryMetadata.suggestedGym("indoor",(-30.0 to -120.0),gyms))
    }
}
