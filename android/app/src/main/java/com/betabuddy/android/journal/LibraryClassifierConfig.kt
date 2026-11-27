package com.betabuddy.android.journal

import kotlinx.serialization.Serializable

/** Generic embedding policy. All domain labels, prototypes and thresholds come from the private pack. */
@Serializable
internal data class LibraryClassifierConfig(
    val schema: Int, val model: String, val samples: Int, val minimumVotes: Int,
    val minimumScore: Double, val minimumMargin: Double, val prototypes: List<Prototype>,
    val androidModel: Model,
    val imageResize: String? = null,
) {
    val imageModes: List<String> get() = if (imageResize == "centerAndFit") listOf("center","fit") else listOf(imageResize ?: "center")
    @Serializable data class Prototype(val tag: String? = null, val vector: List<Double>)
    @Serializable data class Model(val file: String, val sha256: String)
    fun validate() {
        require(schema in 1..2 && (schema == 1 || imageResize != null))
        require(imageResize == null || imageResize in listOf("center","fit","stretch","centerAndFit"))
        require(model == "LibraryImageEncoder" && samples in 3..20 && minimumVotes in 1..samples)
        require(minimumScore.isFinite() && minimumScore in -1.0..1.0 && minimumMargin.isFinite() && minimumMargin in 0.0..2.0)
        val width = prototypes.firstOrNull()?.vector?.size ?: 0
        require(width in 1..4096 && prototypes.any { it.tag == null } && prototypes.any { it.tag != null })
        require(prototypes.all { it.vector.size == width && it.vector.all(Double::isFinite) })
        require(androidModel.file == "library-image-encoder.onnx" && androidModel.sha256.matches(Regex("[0-9a-f]{64}")))
    }
    fun tag(embedding: List<Double>): String? {
        if (embedding.size != prototypes.first().vector.size || !embedding.all(Double::isFinite)) return null
        val ranked = prototypes.map { it.tag to it.vector.zip(embedding).sumOf { (a,b) -> a*b } }.sortedByDescending { it.second }
        val best = ranked.first(); val tag = best.first ?: return null
        val other = ranked.firstOrNull { it.first != tag } ?: return null
        return tag.takeIf { best.second >= minimumScore && best.second - other.second >= minimumMargin }
    }
    fun decision(votes: List<String?>): String? {
        val ranked = votes.filterNotNull().groupingBy { it }.eachCount().entries.sortedByDescending { it.value }
        val first = ranked.firstOrNull() ?: return null
        return first.key.takeIf { first.value >= minimumVotes && (ranked.size == 1 || first.value > ranked[1].value) }
    }
    /** Views of the same frame never create extra temporal evidence. */
    fun frameVote(votes: List<String?>): String? = votes.filterNotNull().toSet().singleOrNull()
}
