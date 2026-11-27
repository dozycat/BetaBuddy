package com.betabuddy.android.data

import java.io.ByteArrayOutputStream
import java.io.File
import java.util.UUID
import java.util.zip.GZIPInputStream
import java.util.zip.GZIPOutputStream
import kotlinx.serialization.encodeToString

/** Candidate observations are loaded only for the record being selected, never the whole library. */
internal object SubjectCandidateStorage {
    fun file(root: File, record: String, reference: String): File {
        UUID.fromString(record); UUID.fromString(reference)
        return File(root, "$record.subjects.$reference.json.gz")
    }

    fun encode(samples: List<SubjectSample>): ByteArray = ByteArrayOutputStream().use { buffer ->
        GZIPOutputStream(buffer).bufferedWriter(Charsets.UTF_8).use { it.write(ClimbStore.json.encodeToString(samples)) }
        buffer.toByteArray()
    }

    fun read(root: File, record: String, reference: String): List<SubjectSample> =
        GZIPInputStream(file(root,record,reference).inputStream()).bufferedReader(Charsets.UTF_8).use {
            ClimbStore.json.decodeFromString(it.readText())
        }
}
